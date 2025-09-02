from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from uuid import uuid4
import json
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

def create_embeddings():
    """Create Bedrock embeddings with error handling."""
    try:
        return BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0')
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock embeddings: {e}")
        raise RuntimeError(f"Bedrock embeddings initialization failed: {e}")

def create_vector_store(embeddings):
    """Create Chroma vector store with error handling."""
    try:
        persist_dir = f"./data/{uuid4()}"
        os.makedirs(persist_dir, exist_ok=True)
        
        return Chroma(
            collection_name="docs_collection",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise RuntimeError(f"Vector store initialization failed: {e}")

def create_text_splitter():
    """Create text splitter with error handling."""
    try:
        return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    except Exception as e:
        logger.error(f"Failed to initialize text splitter: {e}")
        raise RuntimeError(f"Text splitter initialization failed: {e}")

# Initialize components with error handling
try:
    embeddings = create_embeddings()
    vector_store = create_vector_store(embeddings)
    text_splitter = create_text_splitter()
    logger.info("Vector store components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store components: {e}")
    embeddings = None
    vector_store = None
    text_splitter = None