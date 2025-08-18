import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TextractBlock:
    """Represents a Textract block with spatial information"""
    block_id: str
    block_type: str
    text: str
    confidence: float
    bounding_box: Dict[str, float]
    polygon: List[Dict[str, float]]
    page_number: int
    relationships: List[Dict[str, Any]]
    start_char: int  # Position in full document text
    end_char: int    # Position in full document text

@dataclass
class DocumentChunk:
    """Represents a semantic chunk with mapping to source blocks"""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    source_blocks: List[str]  # List of block IDs that contribute to this chunk
    page_numbers: List[int]   # Pages this chunk spans
    bounding_boxes: List[Dict[str, float]]  # Bounding boxes from source blocks
    confidence_scores: List[float]  # Confidence scores from source blocks
    metadata: Dict[str, Any]

class TextractProcessor:
    """Process Textract output and create chunks with bounding box mapping"""
    
    def __init__(self, chunk_size: int = 2000, overlap_size: int = 250):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
    
    def parse_textract_response(self, textract_response: Dict) -> Tuple[str, List[TextractBlock]]:
        """Parse Textract response and extract text with block information"""
        blocks = textract_response.get("Blocks", [])
        
        # Filter for text blocks (LINE, WORD) and sort by reading order
        text_blocks = []
        full_text_parts = []
        current_char_pos = 0
        
        # Get pages for proper ordering
        pages = {}
        for block in blocks:
            if block["BlockType"] == "PAGE":
                pages[block["Id"]] = block.get("Page", 1)
        
        # Process LINE blocks to maintain reading order
        line_blocks = [b for b in blocks if b["BlockType"] == "LINE"]
        line_blocks.sort(key=lambda x: (x.get("Page", 1), 
                                       x["Geometry"]["BoundingBox"]["Top"],
                                       x["Geometry"]["BoundingBox"]["Left"]))
        
        for block in line_blocks:
            text = block.get("Text", "").strip()
            if not text:
                continue
            
            # Create TextractBlock object
            textract_block = TextractBlock(
                block_id=block["Id"],
                block_type=block["BlockType"],
                text=text,
                confidence=block.get("Confidence", 0.0),
                bounding_box=block["Geometry"]["BoundingBox"],
                polygon=block["Geometry"].get("Polygon", []),
                page_number=block.get("Page", 1),
                relationships=block.get("Relationships", []),
                start_char=current_char_pos,
                end_char=current_char_pos + len(text)
            )
            
            text_blocks.append(textract_block)
            full_text_parts.append(text)
            current_char_pos += len(text) + 1  # +1 for space/newline
        
        full_text = " ".join(full_text_parts)
        
        logger.info(f"Parsed Textract response: {len(text_blocks)} blocks, "
                   f"{len(full_text)} characters")
        
        return full_text, text_blocks
    
    def create_chunks_with_block_mapping(self, full_text: str, 
                                       text_blocks: List[TextractBlock]) -> List[DocumentChunk]:
        """Create semantic chunks and map them to source Textract blocks"""
        
        # Create semantic chunks using your existing logic
        chunks = self._create_semantic_chunks(full_text)
        
        # Map each chunk to its source blocks
        document_chunks = []
        
        for i, chunk_text in enumerate(chunks):
            # Find the position of this chunk in the full text
            chunk_start = full_text.find(chunk_text)
            if chunk_start == -1:
                # Fallback: try fuzzy matching or use chunk index * chunk_size
                chunk_start = i * (self.chunk_size - self.overlap_size)
            
            chunk_end = chunk_start + len(chunk_text)
            
            # Find all blocks that overlap with this chunk
            overlapping_blocks = self._find_overlapping_blocks(
                chunk_start, chunk_end, text_blocks
            )
            
            # Extract metadata from overlapping blocks
            source_block_ids = [block.block_id for block in overlapping_blocks]
            page_numbers = list(set(block.page_number for block in overlapping_blocks))
            bounding_boxes = [block.bounding_box for block in overlapping_blocks]
            confidence_scores = [block.confidence for block in overlapping_blocks]
            
            # Calculate chunk-level metadata
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            chunk_metadata = {
                "avg_confidence": avg_confidence,
                "block_count": len(overlapping_blocks),
                "spans_pages": len(page_numbers) > 1,
                "textract_block_types": list(set(block.block_type for block in overlapping_blocks))
            }
            
            document_chunk = DocumentChunk(
                content=chunk_text,
                chunk_index=i,
                start_char=chunk_start,
                end_char=chunk_end,
                source_blocks=source_block_ids,
                page_numbers=sorted(page_numbers),
                bounding_boxes=bounding_boxes,
                confidence_scores=confidence_scores,
                metadata=chunk_metadata
            )
            
            document_chunks.append(document_chunk)
        
        logger.info(f"Created {len(document_chunks)} chunks with block mapping")
        return document_chunks
    
    def _create_semantic_chunks(self, text: str) -> List[str]:
        """Create semantic chunks using your existing text processing logic"""
        # Use your existing TextProcessor logic here
        # This is a simplified version
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence/paragraph boundaries
            if end < len(text):
                # Look back for sentence ending
                for i in range(end, max(start, end - 200), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + 1, end - self.overlap_size)
        
        return chunks
    
    def _find_overlapping_blocks(self, chunk_start: int, chunk_end: int, 
                               text_blocks: List[TextractBlock]) -> List[TextractBlock]:
        """Find blocks that overlap with the given character range"""
        overlapping = []
        
        for block in text_blocks:
            # Check if block overlaps with chunk
            if (block.start_char < chunk_end and block.end_char > chunk_start):
                overlapping.append(block)
        
        return overlapping