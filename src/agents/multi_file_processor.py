import logging
import time
import uuid
from typing import List, Dict, Any, Tuple
from pathlib import Path

import fitz
from agents.enhanced_vector_database import EnhancedVectorDatabase
from agents.textract_processor import TextractProcessor
from ..core.file_handler import FileHandler
from ..core.ocr_engine import OCREngine
from ..core.text_processor import TextProcessor
from ..config.settings import Config
from .multi_file_state import MultiFileDocumentState, FileInfo, ProcessingStatus
from .vector_database import VectorDatabase
from .embedding_service import EmbeddingService
import os
import json
import boto3

RENDERING_DPI = 150

logger = logging.getLogger(__name__)

def upload_multiple_files(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """
    LangGraph node: Handle multiple file uploads (unchanged from your original)
    """
    if state["overall_status"] == ProcessingStatus.SUMMARIZED:
        return state
    try:
        state["overall_status"] = ProcessingStatus.UPLOADING
        state["current_step"] = "upload"
        
        uploaded_files = state.get("uploaded_file_paths", [])
        
        if not uploaded_files:
            raise Exception("No files provided for upload")
        
        file_handler = FileHandler(Config.UPLOAD_DIR, Config.PROCESSED_DIR)
        state["total_files"] = len(uploaded_files)
        state["files_completed"] = 0
        state.update({"files": {}, "file_upload_order": []})
        
        for file_path, file_name in uploaded_files:
            try:
                file_id = str(uuid.uuid4())
                
                is_valid, error_msg = file_handler.validate_file(
                    file_path, Config.MAX_FILE_SIZE, Config.ALLOWED_EXTENSIONS
                )
                
                if not is_valid:
                    logger.error(f"File validation failed for {file_name}: {error_msg}")
                    continue
                
                saved_path, file_type, file_size = file_handler.save_uploaded_file(
                    file_path, file_name
                )
                
                file_info = FileInfo(
                    file_id=file_id,
                    file_name=file_name,
                    file_path=saved_path,
                    file_type=file_type,
                    file_size=file_size,
                    upload_timestamp=time.time(),
                    processing_status=ProcessingStatus.UPLOADED
                )
                
                state["files"][file_id] = file_info
                state["file_upload_order"].append(file_id)
                
                logger.info(f"File uploaded successfully: {file_name} ({file_type})")
                
            except Exception as e:
                logger.error(f"Error uploading file {file_name}: {str(e)}")
                continue
        
        if not state["files"]:
            raise Exception("No files were successfully uploaded")
        
        state["overall_status"] = ProcessingStatus.UPLOADED
        state["processing_progress"] = {"overall": 20}
        
        logger.info(f"Successfully uploaded {len(state['files'])} files")
        return state
        
    except Exception as e:
        logger.error(f"Error in file upload: {str(e)}")
        state["overall_status"] = ProcessingStatus.ERROR
        state["error_message"] = f"File upload failed: {str(e)}"
        return state


def process_all_files_ocr_with_vectors(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """
    LangGraph node: Process OCR for all files AND generate vector embeddings
    """
    try:
        state["overall_status"] = ProcessingStatus.PROCESSING
        state["current_step"] = "ocr_and_vectorization"
        
        files = state["files"]
        if not files:
            raise Exception("No files to process")

        logger.info(f"Processing {len(files)} files with OCR and vectorization")

        # Initialize processors
        ocr_engine = OCREngine(Config.TESSERACT_CONFIG)
        state["max_chunk_size"] = 2000
        state["overlap_size"] = 250
        text_processor = TextProcessor(state["max_chunk_size"], state["overlap_size"])
        
        # Initialize vector database and embedding service
        vector_db = VectorDatabase()
        embedding_service = EmbeddingService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )
        
        def process_single_file_with_vectors(file_id: str, file_info: FileInfo) -> Tuple[str, FileInfo]:
            """Process a single file's OCR and generate embeddings"""
            try:
                logger.info(f"Processing OCR and vectorization for: {file_info.file_name}")
                
                file_info.processing_status = ProcessingStatus.PROCESSING
                
                # Extract text (OCR)
                raw_text, confidence = ocr_engine.extract_text(
                    file_info.file_path, 
                    file_info.file_type
                )
                
                if not raw_text.strip():
                    raise Exception("No text could be extracted")
                
                # Process text and create chunks
                processed_text = text_processor.clean_text(raw_text)
                document_chunks = text_processor.create_chunks(processed_text)
                metadata = text_processor.get_document_metadata(processed_text)
                
                # Generate embeddings for all chunks
                chunk_texts = [chunk["content"] for chunk in document_chunks]
                logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
                
                # Get embeddings in batches for efficiency
                embeddings = embedding_service.get_embeddings_batch(chunk_texts)
                
                # Prepare chunks with embeddings for database insertion
                chunks_with_embeddings = []
                for i, (chunk, embedding) in enumerate(zip(document_chunks, embeddings)):
                    chunk_with_embedding = {
                        'content': chunk['content'],
                        'chunk_index': i,
                        'embedding': embedding,
                        'metadata': {
                            **chunk,  # Original chunk metadata
                            **metadata,  # Document metadata
                            'file_type': file_info.file_type,
                            'file_size': file_info.file_size,
                            'upload_timestamp': file_info.upload_timestamp,
                            'text_quality_score': confidence
                        }
                    }
                    chunks_with_embeddings.append(chunk_with_embedding)
                
                # Store in vector database
                logger.info(f"Storing {len(chunks_with_embeddings)} chunks in vector database...")
                inserted_count = vector_db.insert_document_chunks(
                    file_id=file_id,
                    file_name=file_info.file_name,
                    chunks_with_embeddings=chunks_with_embeddings
                )
                
                # Update file info
                file_info.raw_text = raw_text
                file_info.processed_text = processed_text
                file_info.text_quality_score = confidence
                file_info.document_chunks = [chunk["content"] for chunk in document_chunks]
                file_info.chunk_metadata = document_chunks
                file_info.document_metadata = metadata
                file_info.processing_status = ProcessingStatus.VECTORIZED  # New status
                file_info.vector_chunks_count = inserted_count
                
                logger.info(f"Completed processing for {file_info.file_name}: "
                          f"{len(processed_text)} characters, {inserted_count} vector chunks")
                
                return file_id, file_info
                
            except Exception as e:
                logger.error(f"Error processing {file_info.file_name}: {str(e)}")
                file_info.processing_status = ProcessingStatus.ERROR
                file_info.error_message = str(e)
                return file_id, file_info
        
        # Process files sequentially
        completed_files = 0
        total_files = len(files)
        total_chunks_inserted = 0

        for file_id, file_info in files.items():
            try:
                processed_file_id, updated_file_info = process_single_file_with_vectors(file_id, file_info)

                # Update state with processed file
                state["files"][processed_file_id] = updated_file_info
                completed_files += 1
                
                if hasattr(updated_file_info, 'vector_chunks_count'):
                    total_chunks_inserted += updated_file_info.vector_chunks_count

                # Update progress (20-80% range for this step)
                progress = int((completed_files / total_files) * 60) + 20
                state["processing_progress"]["overall"] = progress
                state["processing_progress"][processed_file_id] = (
                    100 if updated_file_info.processing_status == ProcessingStatus.VECTORIZED else 0
                )
                
            except Exception as e:
                logger.error(f"Error processing file {file_id}: {str(e)}")
        
        # Check if all files were successfully processed
        successful_files = [
            f for f in state["files"].values()
            if f.processing_status == ProcessingStatus.VECTORIZED
        ]

        if len(successful_files) != total_files:
            failed_files = [
                f.file_name for f in state["files"].values()
                if f.processing_status != ProcessingStatus.VECTORIZED
            ]
            raise Exception(f"All files must be successfully processed. Failed files: {', '.join(failed_files)}")

        # Combine all processed text (for backward compatibility)
        combined_texts = []
        combined_chunks = []
        
        for file_info in successful_files:
            if file_info.processed_text:
                combined_texts.append(f"=== {file_info.file_name} ===\n{file_info.processed_text}")
            
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
        state["total_vector_chunks"] = total_chunks_inserted
        state["overall_status"] = ProcessingStatus.VECTORIZED
        state["processing_progress"]["overall"] = 80
        
        # Close database connection
        vector_db.close()
        
        logger.info(f"OCR and vectorization completed. {len(successful_files)} files, "
                   f"{total_chunks_inserted} vector chunks stored.")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in OCR and vectorization: {str(e)}")
        state["overall_status"] = ProcessingStatus.ERROR
        state["error_message"] = f"OCR and vectorization failed: {str(e)}"
        return state


def semantic_search_node(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """
    LangGraph node: Perform semantic search on processed documents
    """
    try:
        query = state.get("search_query", "")
        if not query:
            raise Exception("No search query provided")
        
        logger.info(f"Performing semantic search for: {query}")
        
        # Initialize services
        vector_db = VectorDatabase()
        embedding_service = EmbeddingService(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Get query embedding
        query_embedding = embedding_service.get_embedding(query)
        
        # Search parameters
        limit = state.get("search_limit", 5)
        similarity_threshold = state.get("similarity_threshold", 0.7)
        file_filter = state.get("search_file_filter")  # Optional file_id filter
        
        # Perform similarity search
        search_results = vector_db.similarity_search(
            query_embedding=query_embedding,
            limit=limit,
            file_id=file_filter,
            similarity_threshold=similarity_threshold
        )
        
        # Format results
        formatted_results = []
        for content, filename, chunk_index, metadata, file_id, similarity in search_results:
            formatted_results.append({
                'content': content,
                'filename': filename,
                'chunk_index': chunk_index,
                'file_id': file_id,
                'similarity': similarity,
                'metadata': json.loads(metadata) if metadata else {}
            })
        
        state["search_results"] = formatted_results
        state["search_completed"] = True
        
        # Close database connection
        vector_db.close()
        
        logger.info(f"Semantic search completed. Found {len(formatted_results)} relevant chunks.")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        state["search_error"] = str(e)
        return state


def generate_contextual_response(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """
    LangGraph node: Generate response based on search results
    """
    try:
        search_results = state.get("search_results", [])
        query = state.get("search_query", "")
        
        if not search_results:
            state["response"] = "I couldn't find any relevant information in the uploaded documents for your query."
            return state
        
        # Prepare context from search results
        context_parts = []
        for result in search_results:
            context_parts.append(
                f"From {result['filename']} (chunk {result['chunk_index']}) "
                f"[Similarity: {result['similarity']:.3f}]:\n{result['content']}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate response using your preferred LLM
        # This is a simplified example - integrate with your existing LLM setup
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
        
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        system_prompt = """You are a helpful assistant that answers questions based on document content.
        Answer questions using only the provided context from the documents.
        Always cite which document and section you're referencing.
        If the context doesn't contain enough information, say so clearly."""
        
        user_prompt = f"""
        Based on the following document excerpts, please answer this question: {query}
        
        Document Context:
        {context}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        state["response"] = response.content
        state["response_generated"] = True
        
        # Store the context used for the response
        state["response_context"] = context
        
        logger.info("Generated contextual response based on search results")
        
        return state
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        state["response"] = f"Error generating response: {str(e)}"
        return state
    

def process_textract_files_with_vectors(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """
    LangGraph node: Process Textract files and create vector embeddings with block mapping
    """
    try:
        state["overall_status"] = ProcessingStatus.PROCESSING
        state["current_step"] = "textract_vectorization"
        
        files = state["files"]
        if not files:
            raise Exception("No files to process")

        logger.info(f"Processing {len(files)} files with Textract and vectorization")

        # Initialize processors
        textract_processor = TextractProcessor(
            chunk_size=state.get("max_chunk_size", 2000),
            overlap_size=state.get("overlap_size", 250)
        )
        
        # Initialize vector database and embedding service
        vector_db = EnhancedVectorDatabase()
        embedding_service = EmbeddingService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )
        
        def process_single_textract_file(file_id: str, file_info: FileInfo) -> Tuple[str, FileInfo]:
            """Process a single file's Textract output and generate embeddings"""
            try:
                logger.info(f"Processing Textract data for: {file_info.file_name}")
                
                file_info.processing_status = ProcessingStatus.PROCESSING
                
                # Load Textract response (assuming it's stored in your file_info or loaded separately)
                # You'll need to adapt this based on how you store/access Textract results
                textract_response = _load_textract_response(file_info.file_path)
                
                # Parse Textract response
                full_text, textract_blocks = textract_processor.parse_textract_response(textract_response)
                
                if not full_text.strip():
                    raise Exception("No text could be extracted from Textract response")
                
                # Create chunks with block mapping
                document_chunks = textract_processor.create_chunks_with_block_mapping(
                    full_text, textract_blocks
                )
                
                # Generate embeddings for chunks
                chunk_texts = [chunk.content for chunk in document_chunks]
                logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
                
                embeddings = embedding_service.get_embeddings_batch(chunk_texts)
                
                # Store Textract blocks in database
                logger.info(f"Storing {len(textract_blocks)} Textract blocks...")
                vector_db.insert_textract_blocks(file_id, textract_blocks)
                
                # Store document chunks with embeddings
                logger.info(f"Storing {len(document_chunks)} chunks with embeddings...")
                inserted_chunks = vector_db.insert_document_chunks_with_blocks(
                    file_id=file_id,
                    filename=file_info.file_name,
                    document_chunks=document_chunks,
                    embeddings=embeddings
                )
                
                # Update file info
                file_info.raw_text = full_text
                file_info.processed_text = full_text  # For Textract, these might be the same
                file_info.text_quality_score = sum(block.confidence for block in textract_blocks) / len(textract_blocks)
                file_info.document_chunks = chunk_texts
                file_info.textract_blocks = textract_blocks
                file_info.document_chunks_with_mapping = document_chunks
                file_info.processing_status = ProcessingStatus.VECTORIZED
                file_info.vector_chunks_count = inserted_chunks
                file_info.textract_blocks_count = len(textract_blocks)
                
                logger.info(f"Completed Textract processing for {file_info.file_name}: "
                          f"{len(full_text)} characters, {inserted_chunks} chunks, "
                          f"{len(textract_blocks)} blocks")
                
                return file_id, file_info
                
            except Exception as e:
                logger.error(f"Error processing Textract file {file_info.file_name}: {str(e)}")
                file_info.processing_status = ProcessingStatus.ERROR
                file_info.error_message = str(e)
                return file_id, file_info
        
        def _load_textract_response(self, file_path: str) -> Dict:
            """Load Textract response - adapt based on your storage method"""
            # Option 1: If Textract response is stored as JSON file
            # textract_json_path = file_path.replace('.pdf', '_textract.json')
            # if Path(textract_json_path).exists():
            #     with open(textract_json_path, 'r') as f:
            #         return json.load(f)
            
            # Option 2: If you have Textract response in your state/database
            # return self._get_textract_from_database(file_path)
            
            # Option 3: Call Textract API here if needed
            # return self._call_textract_api(file_path)
            try:
                doc = fitz.open(filename=file_path, filetype="pdf")
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    
                    pix = page.get_pixmap(matrix=fitz.Matrix(RENDERING_DPI/72, RENDERING_DPI/72))
                    image_bytes = pix.tobytes()

                    if len(image_bytes) > 5 * 1024 * 1024:  # 5MB limit
                        print(f"Warning: Page {page_num + 1} image is too large for Textract. Skipping.")
                        continue

                    # page_text = self.extract_text_from_page(image_bytes)
                    textract_client = boto3.client('textract')
                    try:
                        response = textract_client.detect_document_text(
                            Document={'Bytes': image_bytes}
                        )
                        
                        # Extract text from all LINE blocks
                        return response
                    
                    except Exception as e:
                        logger.info(f"Error extracting text with Textract: {e}")
                        return ""
                
                return ""
            except Exception as e:
                logger.error(f"Error reading PDF: {str(e)}")
                raise
            
            raise Exception(f"Textract response not found for {file_path}")
        
        # Process files sequentially
        completed_files = 0
        total_files = len(files)
        total_chunks_inserted = 0
        total_blocks_inserted = 0

        for file_id, file_info in files.items():
            try:
                processed_file_id, updated_file_info = process_single_textract_file(file_id, file_info)

                # Update state
                state["files"][processed_file_id] = updated_file_info
                completed_files += 1
                
                if hasattr(updated_file_info, 'vector_chunks_count'):
                    total_chunks_inserted += updated_file_info.vector_chunks_count
                if hasattr(updated_file_info, 'textract_blocks_count'):
                    total_blocks_inserted += updated_file_info.textract_blocks_count

                # Update progress
                progress = int((completed_files / total_files) * 60) + 20
                state["processing_progress"]["overall"] = progress
                state["processing_progress"][processed_file_id] = (
                    100 if updated_file_info.processing_status == ProcessingStatus.VECTORIZED else 0
                )
                
            except Exception as e:
                logger.error(f"Error processing file {file_id}: {str(e)}")
        
        # Validation and state update
        successful_files = [
            f for f in state["files"].values()
            if f.processing_status == ProcessingStatus.VECTORIZED
        ]

        if len(successful_files) != total_files:
            failed_files = [
                f.file_name for f in state["files"].values()
                if f.processing_status != ProcessingStatus.VECTORIZED
            ]
            raise Exception(f"All files must be successfully processed. Failed files: {', '.join(failed_files)}")

        # Update final state
        state["files_completed"] = len(successful_files)
        state["total_vector_chunks"] = total_chunks_inserted
        state["total_textract_blocks"] = total_blocks_inserted
        state["overall_status"] = ProcessingStatus.VECTORIZED
        state["processing_progress"]["overall"] = 80
        
        vector_db.close()
        
        logger.info(f"Textract vectorization completed. {len(successful_files)} files, "
                   f"{total_chunks_inserted} chunks, {total_blocks_inserted} blocks stored.")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in Textract vectorization: {str(e)}")
        state["overall_status"] = ProcessingStatus.ERROR
        state["error_message"] = f"Textract vectorization failed: {str(e)}"
        return state


def enhanced_semantic_search_node(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """
    LangGraph node: Semantic search with spatial filtering capabilities
    """
    try:
        query = state.get("search_query", "")
        if not query:
            raise Exception("No search query provided")
        
        logger.info(f"Performing enhanced semantic search for: {query}")
        
        # Initialize services
        vector_db = EnhancedVectorDatabase()
        embedding_service = EmbeddingService(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Get query embedding
        query_embedding = embedding_service.get_embedding(query)
        
        # Search parameters
        limit = state.get("search_limit", 5)
        similarity_threshold = state.get("similarity_threshold", 0.7)
        file_filter = state.get("search_file_filter")
        page_filter = state.get("search_page_filter")  # New: filter by pages
        
        # Perform enhanced similarity search
        search_results = vector_db.similarity_search_with_blocks(
            query_embedding=query_embedding,
            limit=limit,
            file_id=file_filter,
            page_filter=page_filter,
            similarity_threshold=similarity_threshold
        )
        
        # Enhanced results with spatial information
        state["search_results"] = search_results
        state["search_completed"] = True
        
        # Additional spatial analysis
        if search_results:
            # Analyze page distribution
            page_distribution = {}
            for result in search_results:
                for page in result['page_numbers']:
                    page_distribution[page] = page_distribution.get(page, 0) + 1
            
            state["search_page_distribution"] = page_distribution
            
            # Calculate average confidence from blocks
            avg_confidence = 0
            total_blocks = 0
            for result in search_results:
                for block in result.get('block_details', []):
                    if block and 'confidence' in block:
                        avg_confidence += block['confidence']
                        total_blocks += 1
            
            if total_blocks > 0:
                state["search_avg_confidence"] = avg_confidence / total_blocks
        
        vector_db.close()
        
        logger.info(f"Enhanced semantic search completed. Found {len(search_results)} relevant chunks "
                   f"across {len(state.get('search_page_distribution', {}))} pages.")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in enhanced semantic search: {str(e)}")
        state["search_error"] = str(e)
        return state