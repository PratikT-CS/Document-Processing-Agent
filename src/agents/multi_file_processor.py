import logging
import time
import uuid
from typing import List, Dict, Any, Tuple
from pathlib import Path
from ..core.file_handler import FileHandler
from ..core.ocr_engine import OCREngine
from ..core.text_processor import TextProcessor
from ..config.settings import Config
from .multi_file_state import MultiFileDocumentState, FileInfo, ProcessingStatus
import boto3
import fitz
from .blueprints import blueprints
import os
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

def upload_multiple_files(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """
    LangGraph node: Handle multiple file uploads
    """
    if state["overall_status"] == ProcessingStatus.SUMMARIZED:
        return state
    try:
        state["overall_status"] = ProcessingStatus.UPLOADING
        state["current_step"] = "upload"
        
        # Get uploaded files from state (would come from Gradio)
        uploaded_files = state.get("uploaded_file_paths", [])  # List of (path, name) tuples
        
        if not uploaded_files:
            raise Exception("No files provided for upload")
        
        # if len(uploaded_files) > state["max_files"]:
        #     raise Exception(f"Too many files. Maximum allowed: {state['max_files']}")
        
        # Initialize file handler
        file_handler = FileHandler(Config.UPLOAD_DIR, Config.PROCESSED_DIR)
        
        print(len(uploaded_files))

        state["total_files"] = len(uploaded_files)
        state["files_completed"] = 0
        state.update({"files": {}, "file_upload_order": []})
        
        # Process each file upload
        for file_path, file_name in uploaded_files:
            try:
                # Generate unique file ID
                file_id = str(uuid.uuid4())
                
                # Validate file
                is_valid, error_msg = file_handler.validate_file(
                    file_path, Config.MAX_FILE_SIZE, Config.ALLOWED_EXTENSIONS
                )
                
                if not is_valid:
                    logger.error(f"File validation failed for {file_name}: {error_msg}")
                    continue
                
                # Save file
                saved_path, file_type, file_size = file_handler.save_uploaded_file(
                    file_path, file_name
                )

                # Upload file to s3 bucket as well for BDA processing
                response = upload_file_to_s3(file_name)

                if not response["uploaded_to_s3"]:
                    raise Exception(f"Failed to upload file {file_name} to S3")
                else:
                    # Create file info
                    file_info = FileInfo(
                        file_id=file_id,
                        file_name=file_name,
                        file_path=saved_path,
                        file_type=file_type,
                        file_size=file_size,
                        upload_timestamp=time.time(),
                        s3_uri=response["s3_uri"],
                        processing_status=ProcessingStatus.UPLOADED
                    )

                # Add to state
                state["files"][file_id] = file_info
                state["file_upload_order"].append(file_id)
                

                logger.info(f"File uploaded successfully: {file_name} ({file_type})")
                
            except Exception as e:
                logger.error(f"Error uploading file {file_name}: {str(e)}")
                continue
        
        if not state["files"]:
            raise Exception("No files were successfully uploaded")
        
        state["overall_status"] = ProcessingStatus.UPLOADED
        state["processing_progress"] = {"overall": 0}
        state["processing_progress"]["overall"] = 20
        
        logger.info(f"Successfully uploaded {len(state['files'])} files")
        return state
        
    except Exception as e:
        logger.error(f"Error in file upload: {str(e)}")
        state["overall_status"] = ProcessingStatus.ERROR
        state["error_message"] = f"File upload failed: {str(e)}"
        return state

def upload_file_to_s3(file_path: str) -> Dict:
    s3_client = boto3.client('s3')
    bucket_name = 'doc-processing-agent-test-k'
    try:
        print(file_path)
        key = f"uploads/{file_path.split('\\')[-1].split('_', 9)[-1]}"
        response = s3_client.put_object(
            Bucket=bucket_name,
            Body=open(file_path, 'rb').read(),
            Key=key,
            ContentType="application/pdf"
        )
        
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            logger.info(f"File {file_path} uploaded to s3 sucessfully")
            return {
                "uploaded_to_s3": True,
                "s3_uri": f"s3://{bucket_name}/{key}"
            }
        else:
            return {
                "uploaded_to_s3": False,
                "s3_uri": None
            }
        
    except Exception as e:
        logger.error(f"Error uploading file to S3: {str(e)}")
        raise

def process_all_files_ocr(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """
    LangGraph node: Process OCR for all files in parallel
    """
    try:
        state["overall_status"] = ProcessingStatus.PROCESSING
        state["current_step"] = "ocr"
        
        files = state["files"]
        if not files:
            raise Exception("No files to process")

        logger.info(f"Total Files: {len(files)}")

        # Initialize processors
        ocr_engine = OCREngine(Config.TESSERACT_CONFIG)
        state["max_chunk_size"] = 2000
        state["overlap_size"] = 250
        text_processor = TextProcessor(state["max_chunk_size"], state["overlap_size"])
        
        def process_single_file(file_id: str, file_info: FileInfo) -> Tuple[str, FileInfo]:
            """Process a single file's OCR"""
            try:
                logger.info(f"Processing OCR for file: {file_info.file_name}")
                
                # Update file status
                file_info.processing_status = ProcessingStatus.PROCESSING
                
                # Extract text
                raw_text, confidence = ocr_engine.extract_text(
                    file_info.file_path, 
                    file_info.file_type
                )
                
                if not raw_text.strip():
                    raise Exception("No text could be extracted")
                
                # Process text
                processed_text = text_processor.clean_text(raw_text)
                document_chunks = text_processor.create_chunks(processed_text)
                metadata = text_processor.get_document_metadata(processed_text)
                
                # Update file info
                file_info.raw_text = raw_text
                file_info.processed_text = processed_text
                file_info.text_quality_score = confidence
                file_info.document_chunks = [chunk["content"] for chunk in document_chunks]
                file_info.chunk_metadata = document_chunks
                file_info.document_metadata = metadata
                file_info.processing_status = ProcessingStatus.OCR_COMPLETE
                
                logger.info(f"OCR completed for {file_info.file_name}: {len(processed_text)} characters")
                
                return file_id, file_info
                
            except Exception as e:
                logger.error(f"Error processing {file_info.file_name}: {str(e)}")
                file_info.processing_status = ProcessingStatus.ERROR
                file_info.error_message = str(e)
                return file_id, file_info
        
        upt_state = process_all_files_via_bda(state)

        logger.info(f"=======================\nState after bda processed: {state}\n=============================\n")

        # Process files sequentially (no ThreadPool)
        completed_files = 0
        total_files = len(files)
        
        for file_id, file_info in files.items():
            try:
                processed_file_id, updated_file_info = process_single_file(file_id, file_info)

                # Update state with processed file
                state["files"][processed_file_id] = updated_file_info
                completed_files += 1

                # Update progress
                progress = int((completed_files / total_files) * 60) + 20  # 20-80% range
                state["processing_progress"]["overall"] = progress
                state["processing_progress"][processed_file_id] = 100 if updated_file_info.processing_status == ProcessingStatus.OCR_COMPLETE else 0
            except Exception as e:
                logger.error(f"Error processing file {file_id}: {str(e)}")
        
        # Require all files to be successfully processed before continuing
        successful_files = [
            f for f in state["files"].values()
            if f.processing_status == ProcessingStatus.OCR_COMPLETE
        ]

        if len(successful_files) != total_files:
            failed_files = [
                f.file_name for f in state["files"].values()
                if f.processing_status != ProcessingStatus.OCR_COMPLETE
            ]
            raise Exception(f"All files must be successfully processed. Failed files: {', '.join(failed_files)}")

        # Combine all processed text
        combined_texts = []
        combined_chunks = []
        
        for file_info in successful_files:
            # Add file text
            if file_info.processed_text:
                combined_texts.append(f"=== {file_info.file_name} ===\n{file_info.processed_text}")
            
            # Add chunks with file metadata
            for i, chunk in enumerate(file_info.document_chunks):
                combined_chunks.append({
                    "content": chunk,
                    "file_id": file_info.file_id,
                    "file_name": file_info.file_name,
                    "file_type": file_info.file_type,
                    "chunk_index": i,
                    "metadata": file_info.chunk_metadata[i] if i < len(file_info.chunk_metadata) else {}
                })
        
        state["combined_text"] = "\n\n".join(combined_texts)
        state["combined_chunks"] = combined_chunks
        state["files_completed"] = len(successful_files)
        state["overall_status"] = ProcessingStatus.OCR_COMPLETE
        state["processing_progress"]["overall"] = 80
        
        logger.info(f"OCR processing completed. {len(successful_files)} files processed successfully.")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in OCR processing: {str(e)}")
        state["overall_status"] = ProcessingStatus.ERROR
        state["error_message"] = f"OCR processing failed: {str(e)}"
        return state

def process_all_files_via_bda(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """Process all  files through BDA (Bedrock Data Automation) for extracting structured data"""
    try:
        state["overall_status"] = ProcessingStatus.PROCESSING
        state["current_step"] = "bda"

        files = state["files"]
        if not files: 
            raise Exception("No files to process")
        
        logger.info(f"Total Files: {len(files)}")

        invocation_arns = []
        files_to_process_as_of_now = ["bill of sale", "compliance pack", "mv-1", "store pack"]
        bucket_name = 'doc-processing-agent-test-k'
        for file_id, file_info in files.items():
            for name in files_to_process_as_of_now:
                if name in file_info.s3_uri.lower():
                    invocation_arns_obj = {name: {}}
                    s3_input_uri = file_info.s3_uri
                    s3_output_uri = f"s3://{bucket_name}/output/{file_info.s3_uri.rsplit('/', 1)[1].replace('.pdf', '')}"
                    response = invoke_bda_job(s3_input_uri, s3_output_uri)
                    invocation_arns_obj[name].update({'invocationArn': response['invocationArn']})

                    invocation_arns.append(invocation_arns_obj)

        logger.info(f"#### BDA Imvoked for all files")

        invocation_results = []

         # Wait for processing to complete
        while len(invocation_results) != len(invocation_arns):
            for document in invocation_arns:
                for key, value in document.items():
                    response = get_invocation_result(value['invocationArn'])
                    if response is not None and response not in invocation_results:
                        invocation_results.append(response)

                        if response['status'] == "Success":
                            try:
                                result = read_json_result_from_s3(response['outputConfiguration']['s3Uri'])
                                result = json.loads(result)
                                # file_info.extracted_data = result["explainability_info"]
                                for file_id, file_info in files.items():
                                    if key in file_info.s3_uri.lower():
                                        file_info.extracted_data = result["explainability_info"]
                                        break
                            except Exception as err:
                                print(f"Error while extracting or saving result: {err}")

        state["overall_status"] = ProcessingStatus.BDA_PROCESSED
        return state
    except Exception as e:
        logger.error(f"Error processing BDA: {e}")
        raise

def filter_blueprint(s3_uri):
    """
    Function to filter blueprint based on the S3 URI.
 
    :param s3_uri: S3 URI of the input file
    :return: Filtered blueprint ARN
    """
    try:
        file_name = s3_uri.lower()
       
        # Define the mapping for file types
        as_of_now_files = ["bill of sale", "compliance pack", "mv-1", "store pack"]
       
        for name in as_of_now_files:
            if name in file_name:
                return blueprints.get(name)
       
        return blueprints.get("mv-1")
   
    except Exception as e:
        logger.error(f"Error in filter_blueprint: {str(e)}")
        # Return default blueprint if error occurs
        return []

def invoke_bda_job(input_uri:str, output_uri:str):
    try: 

        bda_runtime_client = boto3.client(
                "bedrock-data-automation-runtime"
            )

        filtered_blueprint = filter_blueprint(input_uri)
        profile_arn = os.getenv("DATA_AUTOMATION_PROFILE_ARN")
        invoke_params = {
            "dataAutomationProfileArn": profile_arn,  
            "inputConfiguration": {
                "s3Uri": input_uri,
            },
            "outputConfiguration": {
                "s3Uri": output_uri,
            },
            "blueprints": [filtered_blueprint]
        }
        
        response = bda_runtime_client.invoke_data_automation_async(**invoke_params)
        
        return response
 
    except Exception as e:
        logger.error(f"Error in invoke_bda_job: {str(e)}")
        logger.error(f"S3 URI: {input_uri}, Output URI: {output_uri}")
        raise

import time
import boto3

def get_invocation_result(invocation_arn):
    """
    Function to get the result of a BDA job invocation.
    
    :param invocation_arn: ARN of the BDA job invocation
    :return: Result of the BDA job invocation
    """

    bda_runtime_client = boto3.client("bedrock-data-automation-runtime")

    while True:
        response = bda_runtime_client.get_data_automation_status(
            invocationArn=invocation_arn
        )
        status = response.get("status")
        
        if status in ["Success", "ServiceError", "ClientError", "Failed"]:
            break
        
        time.sleep(5)
    return response

import boto3
import json
from urllib.parse import urlparse
import anyio

def read_json_result_from_s3(s3_url: str):
    trimmed_url = s3_url.rsplit('/', 1)[0]
    url = trimmed_url + "/0/custom_output/0/result.json"
    parsed_url = urlparse(url)
    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip('/')
    
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')
    json_data = json.loads(content)

    return json.dumps({"inference_result": json_data["inference_result"], "explainability_info": json_data["explainability_info"]})