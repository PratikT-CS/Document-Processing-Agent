import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    """Handles text processing and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', text)
        
        # Fix common OCR errors (optional)
        text = self._fix_common_ocr_errors(text)
        
        return text.strip()
    
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        if not text:
            return []
        
        chunks = self.text_splitter.split_text(text)
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "content": chunk,
                "chunk_id": i,
                "word_count": len(chunk.split()),
                "char_count": len(chunk)
            })
        
        return chunk_data
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR recognition errors"""
        # Common OCR fixes
        fixes = {
            r'\b0\b': 'O',  # Zero to letter O
            r'\b1\b': 'I',  # One to letter I (context dependent)
            r'\s+': ' ',    # Multiple spaces to single space
        }
        
        for pattern, replacement in fixes.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def get_document_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from document text"""
        if not text:
            return {}
        
        return {
            "word_count": len(text.split()),
            "character_count": len(text),
            "sentence_count": len(re.findall(r'[.!?]+', text)),
            "paragraph_count": len(text.split('\n\n')) if '\n\n' in text else 1,
            "estimated_reading_time": max(1, len(text.split()) // 200)  # ~200 words/minute
        }