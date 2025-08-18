import os
import psycopg2
from pgvector.psycopg2 import register_vector
from typing import List, Tuple, Optional, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, database_url: str = None):
        """Initialize vector database connection"""
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.database_url)
            register_vector(self.conn)
            self.conn.autocommit = True
            logger.info("Connected to vector database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def ensure_connection(self):
        """Ensure database connection is alive"""
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT 1")
        except:
            logger.warning("Database connection lost, reconnecting...")
            self._connect()
    
    def insert_document_chunks(self, file_id: str, file_name: str, 
                              chunks_with_embeddings: List[Dict[str, Any]]) -> int:
        """Insert multiple document chunks with embeddings in batch"""
        self.ensure_connection()
        cur = self.conn.cursor()
        
        try:
            inserted_count = 0
            for chunk_data in chunks_with_embeddings:
                cur.execute("""
                    INSERT INTO documents 
                    (file_id, filename, content, chunk_index, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    file_id,
                    file_name,
                    chunk_data['content'],
                    chunk_data['chunk_index'],
                    chunk_data['embedding'],
                    json.dumps(chunk_data.get('metadata', {}))
                ))
                inserted_count += 1
            
            logger.info(f"Inserted {inserted_count} chunks for file: {file_name}")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error inserting chunks for {file_name}: {e}")
            raise
    
    def similarity_search(self, query_embedding: List[float], 
                         limit: int = 5, file_id: str = None, 
                         similarity_threshold: float = 0.0) -> List[Tuple]:
        """Search for similar document chunks"""
        self.ensure_connection()
        cur = self.conn.cursor()
        
        try:
            if file_id:
                cur.execute("""
                    SELECT content, filename, chunk_index, metadata, file_id,
                           1 - (embedding <=> %s) as similarity
                    FROM documents 
                    WHERE file_id = %s AND (1 - (embedding <=> %s)) >= %s
                    ORDER BY embedding <=> %s 
                    LIMIT %s
                """, (query_embedding, file_id, query_embedding, 
                      similarity_threshold, query_embedding, limit))
            else:
                cur.execute("""
                    SELECT content, filename, chunk_index, metadata, file_id,
                           1 - (embedding <=> %s) as similarity
                    FROM documents 
                    WHERE (1 - (embedding <=> %s)) >= %s
                    ORDER BY embedding <=> %s 
                    LIMIT %s
                """, (query_embedding, query_embedding, similarity_threshold, 
                      query_embedding, limit))
            
            return cur.fetchall()
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_document_files(self) -> List[Dict[str, Any]]:
        """Get list of all document files with metadata"""
        self.ensure_connection()
        cur = self.conn.cursor()
        
        cur.execute("""
            SELECT file_id, filename, COUNT(*) as chunk_count,
                   MIN(created_at) as upload_date
            FROM documents 
            GROUP BY file_id, filename
            ORDER BY upload_date DESC
        """)
        
        return [
            {
                'file_id': row[0],
                'filename': row[1], 
                'chunk_count': row[2],
                'upload_date': row[3]
            }
            for row in cur.fetchall()
        ]
    
    def delete_document_by_file_id(self, file_id: str) -> int:
        """Delete all chunks for a specific file"""
        self.ensure_connection()
        cur = self.conn.cursor()
        
        cur.execute("DELETE FROM documents WHERE file_id = %s", (file_id,))
        return cur.rowcount
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()