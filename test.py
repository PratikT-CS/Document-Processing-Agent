import os
import json
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document

# ----------------------------
# 1. Setup API Key (if using OpenAI embeddings)
# ----------------------------
# Make sure you have set OPENAI_API_KEY in your environment
# Example (Windows PowerShell):
#   $env:OPENAI_API_KEY="sk-xxxx"
# Example (Linux/Mac):
#   export OPENAI_API_KEY="sk-xxxx"


embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

# Initialize Chroma (set persist_directory if you want to save vectors)
vectorstore = Chroma(
    collection_name="doc_chunks",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# ----------------------------
# 2. Sample Data: key-value with bounding box metadata
# ----------------------------
texts_with_metadata = [
    {
        "key": "Policy Number",
        "value": "6F7G8H9I0J",
        "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 230}
    },
    {
        "key": "Insured Name",
        "value": "John Doe",
        "bbox": {"x1": 120, "y1": 250, "x2": 400, "y2": 280}
    },
    {
        "key": "Premium Amount",
        "value": "$500",
        "bbox": {"x1": 150, "y1": 300, "x2": 250, "y2": 330}
    }
]

# Convert into LangChain Documents
documents = []
for item in texts_with_metadata:
    text = f"{item['key']}: {item['value']}"
    metadata = {"bbox": json.dumps(item["bbox"]), "key": item["key"]}
    documents.append(Document(page_content=text, metadata=metadata))

# Add documents to vectorstore
vectorstore.add_documents(documents)

# ----------------------------
# 3. Create Retriever
# ----------------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# ----------------------------
# 4. Run Retrieval
# ----------------------------
query = "What is the policy number?"
results = retriever.get_relevant_documents(query)

print("\nQuery:", query)
print("\nTop Matches:")
for r in results:
    print("-", r.page_content, "| Metadata:", r.metadata)


# Initialize Chroma (set persist_directory if you want to save vectors)
vectorstore = Chroma(
    collection_name="doc_chunks",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# ----------------------------
# 2. Sample Data: key-value with bounding box metadata
# ----------------------------
texts_with_metadata = [
    {
        "key": "Policy Number",
        "value": "6F7G8H9I0J",
        "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 230}
    },
    {
        "key": "Insured Name",
        "value": "John Doe",
        "bbox": {"x1": 120, "y1": 250, "x2": 400, "y2": 280}
    },
    {
        "key": "Premium Amount",
        "value": "$500",
        "bbox": {"x1": 150, "y1": 300, "x2": 250, "y2": 330}
    }
]

# Convert into LangChain Documents
documents = []
for item in texts_with_metadata:
    text = f"{item['key']}: {item['value']}"
    metadata = {"bbox": item["bbox"], "key": item["key"]}
    documents.append(Document(page_content=text, metadata=metadata))

# Add documents to vectorstore
vectorstore.add_documents(documents)

# ----------------------------
# 3. Create Retriever
# ----------------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# ----------------------------
# 4. Run Retrieval
# ----------------------------
query = "What is the policy number?"
results = retriever.get_relevant_documents(query)

print("\nQuery:", query)
print("\nTop Matches:")
for r in results:
    print("-", r.page_content, "| Metadata:", r.metadata)
