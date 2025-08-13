import logging
from typing import Dict, Any
from ..core.ocr_engine import OCREngine
from ..core.text_processor import TextProcessor
from ..config.settings import Config
from .graph_state import DocumentProcessingState

logger = logging.getLogger(__name__)

def extract_text_from_document(state: DocumentProcessingState) -> DocumentProcessingState:
    """
    LangGraph node: Extract text from uploaded document
    """
    try:
        # Update processing status
        state["processing_status"] = "processing"
        state["current_step"] = "extract"
        state["processing_progress"] = 50
        
        # Get file info from state
        file_path = state["uploaded_file_path"]
        file_type = "pdf"
        
        if not file_path or not file_type:
            raise Exception("File path or type not found in state")
        
        # Initialize processors
        ocr_engine = OCREngine(Config.TESSERACT_CONFIG)
        text_processor = TextProcessor(Config.MAX_CHUNK_SIZE, Config.CHUNK_OVERLAP)
        
        # Extract text
        logger.info(f"Extracting text from {file_type} file: {file_path}")
        raw_text, confidence = ocr_engine.extract_text(file_path, file_type)
        
        if not raw_text.strip():
            raise Exception("No text could be extracted from the document")
        
        # Process text
        processed_text = text_processor.clean_text(raw_text)
        document_chunks = text_processor.create_chunks(processed_text)
        metadata = text_processor.get_document_metadata(processed_text)
        
        # Update state
        state["raw_text"] = raw_text
        state["processed_text"] = processed_text
        state["text_quality_score"] = confidence
        state["document_chunks"] = [chunk["content"] for chunk in document_chunks]
        state["chunk_metadata"] = document_chunks
        state["document_metadata"] = metadata
        state["processing_progress"] = 75
        
        logger.info(f"Raw Text: \n{raw_text}")
        logger.info(f"Processed Text: \n{processed_text}")
        logger.info(f"Chunks: \n{document_chunks}")

        logger.info(f"Text extraction completed. Extracted {len(processed_text)} characters")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in text extraction: {str(e)}")
        state["processing_status"] = "error"
        state["error_message"] = f"Text extraction failed: {str(e)}"
        return state