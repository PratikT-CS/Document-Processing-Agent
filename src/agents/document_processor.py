import logging
from typing import Dict, Any
from ..core.file_handler import FileHandler
from ..config.settings import Config
from .graph_state import DocumentProcessingState

logger = logging.getLogger(__name__)

def process_uploaded_file(state: DocumentProcessingState) -> DocumentProcessingState:
    """
    LangGraph node: Process uploaded file and update state
    """
    try:
        # Update processing status
        state["processing_status"] = "processing"
        state["current_step"] = "upload"
        state["processing_progress"] = 10
        
        # Initialize file handler
        file_handler = FileHandler(Config.UPLOAD_DIR, Config.PROCESSED_DIR)
        
        # Get file path from state (this would come from Gradio upload)
        uploaded_file_path = state.get("uploaded_file_path")
        if not uploaded_file_path:
            raise Exception("No file path provided")
        
        original_filename = state.get("file_name", "unknown_file")
        
        # Validate file
        is_valid, error_msg = file_handler.validate_file(
            uploaded_file_path, 
            Config.MAX_FILE_SIZE, 
            Config.ALLOWED_EXTENSIONS
        )
        
        if not is_valid:
            state["processing_status"] = "error"
            state["error_message"] = error_msg
            return state
        
        # Save file
        saved_path, file_type, file_size = file_handler.save_uploaded_file(
            uploaded_file_path, original_filename
        )
        
        # Update state
        state["uploaded_file_path"] = saved_path
        state["file_type"] = file_type
        state["file_size"] = file_size
        state["processing_status"] = "uploaded"
        state["processing_progress"] = 25
        state["error_message"] = None
        
        logger.info(f"File processed successfully")
    except Exception as e:
        return f"Error: {e}"