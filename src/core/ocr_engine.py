import pytesseract
import cv2
import numpy as np
from PIL import Image
import PyPDF2
import logging
from typing import Tuple, Optional
from pathlib import Path
import boto3
import fitz

logger = logging.getLogger(__name__)
RENDERING_DPI = 150

class OCREngine:
    """Handles text extraction from various document types"""
    
    def __init__(self, tesseract_config: str = '--oem 3 --psm 6'):
        self.tesseract_config = tesseract_config
    
    def extract_text(self, file_path: str, file_type: str) -> Tuple[str, float]:
        """
        Extract text from file based on type
        Returns: (extracted_text, confidence_score)
        """
        try:
            if file_type == 'pdf':
                return self._extract_from_pdf(file_path)
            elif file_type == 'image':
                return self._extract_from_image(file_path)
            elif file_type == 'docx':
                return self._extract_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise
    
    def extract_text_from_page(self, page_image_bytes: bytes) -> str:
        textract_client = boto3.client('textract')
        try:
            response = textract_client.detect_document_text(
                Document={'Bytes': page_image_bytes}
            )
            
            # Extract text from all LINE blocks
            text_lines = []
            for block in response['Blocks']:
                if block['BlockType'] == 'LINE':
                    text_lines.append(block['Text'])
            
            return "\n".join(text_lines)
        
        except Exception as e:
            logger.info(f"Error extracting text with Textract: {e}")
            return ""

    def _extract_from_pdf(self, file_path: str) -> Tuple[str, float]:
        """Extract text from PDF file"""
        text_content = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                    else:
                        # If no text found, might be scanned PDF - use OCR
                        logger.warning(f"No text found on page {page_num + 1}, might need OCR")
            
            full_text = '\n'.join(text_content)
            confidence = 1.0 if full_text.strip() else 0.0  # PDF text extraction is reliable
            
            return full_text, confidence
            
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            raise
        
        # try:
        #     doc = fitz.open(stream=file_path, filetype="pdf")
        #     for page_num in range(doc.page_count):
        #         page = doc.load(page_num)
                
        #         pix = page.get_pixmap(matrix=fitz.Matrix(RENDERING_DPI/72, RENDERING_DPI/72))
        #         image_bytes = pix.tobytes()

        #         if len(image_bytes) > 5 * 1024 * 1024:  # 5MB limit
        #             print(f"Warning: Page {page_num + 1} image is too large for Textract. Skipping.")
        #             continue

        #         page_text = self.extract_text_from_page(image_bytes)

        #         if page_text.strip():
        #                 text_content.append(page_text)
        #         else:
        #             # If no text found, might be scanned PDF - use OCR
        #             logger.warning(f"No text found on page {page_num + 1}")

        #     full_text = '\n'.join(text_content)
        #     confidence = 1.0 if full_text.strip() else 0.0  # PDF text extraction is reliable
            
        #     return full_text, confidence
        # except Exception as e:
        #     logger.error(f"Error reading PDF: {str(e)}")
        #     raise

    def _extract_from_image(self, file_path: str) -> Tuple[str, float]:
        """Extract text from image using OCR"""
        try:
            # Load and preprocess image
            image = cv2.imread(file_path)
            processed_image = self._preprocess_image(image)
            
            # Perform OCR
            ocr_data = pytesseract.image_to_data(
                processed_image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and calculate confidence
            text_parts = []
            confidences = []
            
            for i in range(len(ocr_data['text'])):
                if int(ocr_data['conf'][i]) > 0:  # Valid confidence
                    text = ocr_data['text'][i].strip()
                    if text:
                        text_parts.append(text)
                        confidences.append(float(ocr_data['conf'][i]))
            
            full_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            return full_text, avg_confidence
            
        except Exception as e:
            logger.error(f"Error performing OCR: {str(e)}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh