import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # File settings
    UPLOAD_DIR = "data/uploads"
    PROCESSED_DIR = "data/processed"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.docx'}
    
    # OCR settings
    TESSERACT_CONFIG = '--oem 3 --psm 6'
    OCR_LANGUAGES = ['en']  # Add more languages as needed
    
    # Text processing
    MAX_CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # LLM settings
    MODEL_NAME = "google_genai:gemini-2.5-flash"
    TEMPERATURE = 0.7
    MAX_TOKENS = 1500
    
    # Gradio settings
    GRADIO_PORT = 7868
    GRADIO_SHARE = False
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)