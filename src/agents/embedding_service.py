# embedding_service.py
import openai
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        """Initialize embedding service"""
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.dimension = 1536 if model == "text-embedding-ada-002" else 1536
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Get embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error in batch embedding {i//batch_size + 1}: {e}")
                raise
        
        return embeddings