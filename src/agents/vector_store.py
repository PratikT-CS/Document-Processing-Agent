from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from uuid import uuid4
import json

embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0')

vector_store = Chroma(
    collection_name="docs_collection",
    embedding_function=embeddings,
    persist_directory=f"./data/{uuid4()}",
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)