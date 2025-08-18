# Enhanced vector database schema
def create_enhanced_vector_schema():
    """Enhanced database schema to support Textract block mapping"""
    return """
    -- Main documents table for chunks
    CREATE TABLE IF NOT EXISTS document_chunks (
        id SERIAL PRIMARY KEY,
        file_id VARCHAR(255) NOT NULL,
        filename VARCHAR(255) NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding vector(1536),
        start_char INTEGER NOT NULL,
        end_char INTEGER NOT NULL,
        page_numbers INTEGER[] NOT NULL,
        source_block_ids TEXT[] NOT NULL,
        chunk_metadata JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Separate table for Textract blocks with full spatial metadata
    CREATE TABLE IF NOT EXISTS textract_blocks (
        id SERIAL PRIMARY KEY,
        file_id VARCHAR(255) NOT NULL,
        block_id VARCHAR(255) NOT NULL,
        block_type VARCHAR(50) NOT NULL,
        text TEXT,
        confidence FLOAT,
        bounding_box JSONB NOT NULL,
        polygon JSONB,
        page_number INTEGER NOT NULL,
        relationships JSONB,
        start_char INTEGER,
        end_char INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
    
    CREATE INDEX IF NOT EXISTS document_chunks_file_id_idx ON document_chunks(file_id);
    CREATE INDEX IF NOT EXISTS textract_blocks_file_id_idx ON textract_blocks(file_id);
    CREATE INDEX IF NOT EXISTS textract_blocks_block_id_idx ON textract_blocks(block_id);
    CREATE INDEX IF NOT EXISTS textract_blocks_page_idx ON textract_blocks(page_number);
    
    -- View to get chunk details with block information
    CREATE OR REPLACE VIEW chunk_with_blocks AS
    SELECT 
        dc.*,
        tb.block_id,
        tb.bounding_box,
        tb.confidence as block_confidence,
        tb.page_number
    FROM document_chunks dc
    CROSS JOIN UNNEST(dc.source_block_ids) AS block_id
    LEFT JOIN textract_blocks tb ON tb.block_id = block_id AND tb.file_id = dc.file_id;
    """

from typing import List, Dict, Any
import json
import logging

from agents.vector_database import VectorDatabase

logger = logging.getLogger(__name__)

from agents.textract_processor import DocumentChunk, TextractBlock

# enhanced_vector_database.py
class EnhancedVectorDatabase(VectorDatabase):
    """Extended VectorDatabase with Textract block support"""
    
    def insert_textract_blocks(self, file_id: str, textract_blocks: List[TextractBlock]) -> int:
        """Insert Textract blocks with full spatial metadata"""
        self.ensure_connection()
        cur = self.conn.cursor()
        
        try:
            inserted_count = 0
            for block in textract_blocks:
                cur.execute("""
                    INSERT INTO textract_blocks 
                    (file_id, block_id, block_type, text, confidence, 
                     bounding_box, polygon, page_number, relationships, start_char, end_char)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    file_id,
                    block.block_id,
                    block.block_type,
                    block.text,
                    block.confidence,
                    json.dumps(block.bounding_box),
                    json.dumps(block.polygon),
                    block.page_number,
                    json.dumps(block.relationships),
                    block.start_char,
                    block.end_char
                ))
                inserted_count += 1
            
            logger.info(f"Inserted {inserted_count} Textract blocks")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error inserting Textract blocks: {e}")
            raise
    
    def insert_document_chunks_with_blocks(self, file_id: str, filename: str, 
                                         document_chunks: List[DocumentChunk],
                                         embeddings: List[List[float]]) -> int:
        """Insert document chunks with block mapping and embeddings"""
        self.ensure_connection()
        cur = self.conn.cursor()
        
        try:
            inserted_count = 0
            for chunk, embedding in zip(document_chunks, embeddings):
                cur.execute("""
                    INSERT INTO document_chunks 
                    (file_id, filename, chunk_index, content, embedding, 
                     start_char, end_char, page_numbers, source_block_ids, chunk_metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    file_id,
                    filename,
                    chunk.chunk_index,
                    chunk.content,
                    embedding,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.page_numbers,
                    chunk.source_blocks,
                    json.dumps(chunk.metadata)
                ))
                inserted_count += 1
            
            logger.info(f"Inserted {inserted_count} document chunks with block mapping")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error inserting document chunks: {e}")
            raise
    
    def similarity_search_with_blocks(self, query_embedding: List[float], 
                                    limit: int = 5, file_id: str = None,
                                    page_filter: List[int] = None,
                                    similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Enhanced similarity search that includes block information"""
        self.ensure_connection()
        cur = self.conn.cursor()
        
        try:
            # Base query
            where_conditions = ["(1 - (dc.embedding <=> %s)) >= %s"]
            params = [query_embedding, similarity_threshold]
            
            # Add filters
            if file_id:
                where_conditions.append("dc.file_id = %s")
                params.append(file_id)
            
            if page_filter:
                where_conditions.append("dc.page_numbers && %s")
                params.append(page_filter)
            
            where_clause = " AND ".join(where_conditions)
            
            # Query with block information
            cur.execute(f"""
                SELECT 
                    dc.content,
                    dc.filename,
                    dc.chunk_index,
                    dc.file_id,
                    dc.page_numbers,
                    dc.source_block_ids,
                    dc.chunk_metadata,
                    1 - (dc.embedding <=> %s) as similarity,
                    array_agg(
                        json_build_object(
                            'block_id', tb.block_id,
                            'bounding_box', tb.bounding_box,
                            'confidence', tb.confidence,
                            'page_number', tb.page_number
                        )
                    ) as block_details
                FROM document_chunks dc
                LEFT JOIN textract_blocks tb ON tb.block_id = ANY(dc.source_block_ids) 
                    AND tb.file_id = dc.file_id
                WHERE {where_clause}
                GROUP BY dc.id, dc.content, dc.filename, dc.chunk_index, dc.file_id, 
                         dc.page_numbers, dc.source_block_ids, dc.chunk_metadata, dc.embedding
                ORDER BY similarity DESC
                LIMIT %s
            """, params + [query_embedding, limit])
            
            results = []
            for row in cur.fetchall():
                result = {
                    'content': row[0],
                    'filename': row[1],
                    'chunk_index': row[2],
                    'file_id': row[3],
                    'page_numbers': row[4],
                    'source_block_ids': row[5],
                    'chunk_metadata': json.loads(row[6]) if row[6] else {},
                    'similarity': row[7],
                    'block_details': row[8] if row[8] else []
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in enhanced similarity search: {e}")
            return []
    
    def get_blocks_for_chunk(self, file_id: str, chunk_id: int) -> List[Dict[str, Any]]:
        """Get all Textract blocks for a specific chunk"""
        self.ensure_connection()
        cur = self.conn.cursor()
        
        cur.execute("""
            SELECT tb.*
            FROM document_chunks dc
            JOIN textract_blocks tb ON tb.block_id = ANY(dc.source_block_ids) 
                AND tb.file_id = dc.file_id
            WHERE dc.file_id = %s AND dc.chunk_index = %s
            ORDER BY tb.page_number, tb.start_char
        """, (file_id, chunk_id))
        
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]