from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage

class DocumentProcessingState(TypedDict):
    """State schema for the document processing LangGraph workflow"""
    
    # File information
    uploaded_file_path: Optional[str]
    file_type: Optional[str]  # "pdf", "image", "docx"
    file_name: Optional[str]
    file_size: Optional[int]
    
    # Processing status
    processing_status: str  # "idle", "uploaded", "processing", "summarized", "ready", "error"
    current_step: str  # "upload", "extract", "summarize", "chat"
    error_message: Optional[str]
    processing_progress: int  # 0-100
    
    # Document content
    raw_text: Optional[str]
    processed_text: Optional[str]
    text_quality_score: Optional[float]  # OCR confidence if applicable
    
    # Summary and questions
    document_summary: Optional[str]
    suggested_questions: List[str]
    document_metadata: Dict[str, Any]  # page count, word count, etc.
    
    # Chat functionality
    messages: List[BaseMessage]
    current_query: Optional[str]
    response: Optional[str]
    chat_history: List[Dict[str, str]]  # {"role": "user/assistant", "content": "..."}
    
    # Context for retrieval
    document_chunks: List[str]
    relevant_chunks: List[str]
    chunk_metadata: List[Dict[str, Any]]  # chunk info like page numbers, positions
    
    # Configuration
    max_chunk_size: int
    overlap_size: int
    temperature: float
    
    def __init__(self):
        # Set default values
        return {
            "uploaded_file_path": None,
            "file_type": None,
            "file_name": None,
            "file_size": None,
            "processing_status": "idle",
            "current_step": "upload",
            "error_message": None,
            "processing_progress": 0,
            "raw_text": None,
            "processed_text": None,
            "text_quality_score": None,
            "document_summary": None,
            "suggested_questions": [],
            "document_metadata": {},
            "messages": [],
            "current_query": None,
            "response": None,
            "chat_history": [],
            "document_chunks": [],
            "relevant_chunks": [],
            "chunk_metadata": [],
            "max_chunk_size": 1000,
            "overlap_size": 200,
            "temperature": 0.7
        }