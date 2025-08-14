from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from dataclasses import dataclass
from enum import Enum

class ProcessingStatus(Enum):
    IDLE = "idle"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    OCR_COMPLETE = "ocr_complete"
    SUMMARIZED = "summarized"
    READY = "ready"
    ERROR = "error"

@dataclass
class FileInfo:
    """Information about a single file"""
    file_id: str
    file_name: str
    file_path: str
    file_type: str  # "pdf", "image", "docx"
    file_size: int
    upload_timestamp: float
    
    # Processing status for this file
    processing_status: ProcessingStatus
    error_message: Optional[str] = None
    
    # Extracted content
    raw_text: Optional[str] = None
    processed_text: Optional[str] = None
    text_quality_score: Optional[float] = None
    
    # Document chunks and metadata
    document_chunks: List[str] = None
    chunk_metadata: List[Dict[str, Any]] = None
    document_metadata: Dict[str, Any] = None
    
    # Individual file summary
    individual_summary: Optional[str] = None
    
    def __post_init__(self):
        if self.document_chunks is None:
            self.document_chunks = []
        if self.chunk_metadata is None:
            self.chunk_metadata = []
        if self.document_metadata is None:
            self.document_metadata = {}

class MultiFileDocumentState(TypedDict):
    """Extended state schema for multi-file document processing"""

    uploaded_file_paths: List[str]

    # Overall processing status
    overall_status: ProcessingStatus
    current_step: str  # "upload", "ocr", "summarize", "chat"
    processing_progress: Dict[str, int]  # Progress per file and overall
    error_message: Optional[str]
    
    # File management
    files: Dict[str, FileInfo]  # file_id -> FileInfo
    file_upload_order: List[str]  # Order of file uploads
    total_files: int
    files_completed: int
    
    # Combined document content (after OCR completion)
    combined_text: Optional[str]
    combined_chunks: List[Dict[str, Any]]  # chunks with file_id metadata
    combined_summary: Optional[str]
    suggested_questions: List[str]
    
    # Cross-document analysis
    document_relationships: Dict[str, Any]  # Relationships between documents
    topic_analysis: Dict[str, Any]  # Common topics across documents
    
    # Chat functionality
    messages: List[BaseMessage]
    current_query: Optional[str]
    response: Optional[str]
    chat_history: List[Dict[str, str]]
    
    # Context for retrieval (combines all files)
    relevant_chunks: List[Dict[str, Any]]  # chunks with source file info
    
    # Configuration
    max_chunk_size: int
    overlap_size: int
    temperature: float
    max_files: int
    
    def __init__(self):
        return {
            "overall_status": ProcessingStatus.IDLE,
            "current_step": "upload",
            "processing_progress": Dict({"overall": 0}),
            "error_message": None,
            "files": {},
            "file_upload_order": [],
            "total_files": 0,
            "files_completed": 0,
            "combined_text": None,
            "combined_chunks": [],
            "combined_summary": None,
            "suggested_questions": [],
            "document_relationships": {},
            "topic_analysis": {},
            "messages": [],
            "current_query": None,
            "response": None,
            "chat_history": [],
            "relevant_chunks": [],
            "max_chunk_size": 1000,
            "overlap_size": 200,
            "temperature": 0.7,
            "max_files": 10
        }