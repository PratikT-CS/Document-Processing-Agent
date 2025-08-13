import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
import mimetypes

class FileHandler:
    """Handles file operations for document processing"""
    
    def __init__(self, upload_dir: str, processed_dir: str):
        self.upload_dir = Path(upload_dir)
        self.processed_dir = Path(processed_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def save_uploaded_file(self, file_path: str, original_filename: str) -> Tuple[str, str, int]:
        """
        Save uploaded file to upload directory
        Returns: (saved_path, file_type, file_size)
        """
        try:
            # Generate unique filename
            file_extension = Path(original_filename).suffix.lower()
            safe_filename = self._generate_safe_filename(original_filename)
            
            # Determine destination path
            dest_path = self.upload_dir / safe_filename
            
            # Copy file
            shutil.copy2(file_path, dest_path)
            
            # Get file info
            file_size = dest_path.stat().st_size
            file_type = self._determine_file_type(dest_path)
            
            return str(dest_path), file_type, file_size
            
        except Exception as e:
            raise Exception(f"Error saving file: {str(e)}")
    
    def _generate_safe_filename(self, filename: str) -> str:
        """Generate a safe filename"""
        import time
        import re
        
        # Remove unsafe characters
        safe_name = re.sub(r'[^\w\-_\.]', '_', filename)
        # Add timestamp to avoid conflicts
        name, ext = os.path.splitext(safe_name)
        timestamp = str(int(time.time()))
        return f"{name}_{timestamp}{ext}"
    
    def _determine_file_type(self, file_path: Path) -> str:
        """Determine file type from extension and MIME type"""
        extension = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        if extension == '.pdf':
            return 'pdf'
        elif extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return 'image'
        elif extension == '.docx':
            return 'docx'
        else:
            return 'unknown'
    
    def validate_file(self, file_path: str, max_size: int, allowed_extensions: set) -> Tuple[bool, str]:
        """
        Validate uploaded file
        Returns: (is_valid, error_message)
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            return False, "File does not exist"
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > max_size:
            return False, f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size"
        
        # Check extension
        if path.suffix.lower() not in allowed_extensions:
            return False, f"File type {path.suffix} not supported"
        
        return True, ""