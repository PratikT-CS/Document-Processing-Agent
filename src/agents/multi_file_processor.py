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
                
                # Create file info
                file_info = FileInfo(
                    file_id=file_id,
                    file_name=file_name,
                    file_path=saved_path,
                    file_type=file_type,
                    file_size=file_size,
                    upload_timestamp=time.time(),
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
        logger.info(f"+++++++++++++++++++++++++++++++Files Completed: {state['files_completed']}")
        state["overall_status"] = ProcessingStatus.OCR_COMPLETE
        state["processing_progress"]["overall"] = 80
        
        logger.info(f"OCR processing completed. {len(successful_files)} files processed successfully.")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in OCR processing: {str(e)}")
        state["overall_status"] = ProcessingStatus.ERROR
        state["error_message"] = f"OCR processing failed: {str(e)}"
        return state