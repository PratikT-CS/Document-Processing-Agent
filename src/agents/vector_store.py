from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma

embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0')

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./data/embeddings",  # Where to save data locally, remove if not necessary
)

