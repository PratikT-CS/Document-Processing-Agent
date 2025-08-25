import logging
import time
import uuid
from typing import Dict, List, Any
from .vector_store import vector_store, text_splitter
import fitz
import os
from .multi_file_state import MultiFileDocumentState, FileInfo, ProcessingStatus
from langchain_core.documents import Document
import json

logger = logging.getLogger(__name__)

def flatten_kv_items(data):
    final_data = []
    for item in data:
        for key, value in item.items():
            if isinstance(value, dict) and "success" in value.keys():
                final_data.append({key: value})
            else:
                for skey, svalue in value.items():
                    final_data.append({skey: svalue})
    return final_data

def store_embeddings(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """
    store embeddings of documents and visual outputs in vector store with metadata.
    """ 
    if state["overall_status"] == ProcessingStatus.VECTORIZED:
        return state
    try:
        files = state["files"]
        docs = []
        
        for file_id, file_info in files.items():
            docs.append(Document(
                page_content=file_info.processed_text,
                id=str(uuid.uuid4()),
                metadata={
                    "type": "text",
                    "source": file_info.file_name,
                    "bounding_box": json.dumps({}),
                    "page": 0,
                    "key": "",
                    "value": "",
                    "type-value": ""
                }
            ))
            
        docs_splits = text_splitter.split_documents(docs)
        
        for file_id, file_info in files.items():    
            flatten_extracted_data = flatten_kv_items(file_info.extracted_data)
            
            all_extracted_fields = {}
            for item in flatten_extracted_data:
                for key, value in item.items():
                    if isinstance(value, list):
                        continue
                    if "geometry" in value.keys() and len(value["geometry"]) > 0:
                        all_extracted_fields[key] = value["value"]
                        docs_splits.append(Document(
                            page_content=f"{key} is {value['value'] if not isinstance(value['value'], bool) else 'present' if value['value'] == True else 'not present'}",
                            id=str(uuid.uuid4()),
                            metadata={
                                "type": "key-value",
                                "source": file_info.file_name,
                                "bounding_box": json.dumps(value["geometry"][0]["boundingBox"]),
                                "page": value["geometry"][0]["page"],
                                "key": str(key),
                                "value": value["value"],
                                "type-value": value["type"]
                            }
                        ))
        
            file_info.extracted_data_structured = all_extracted_fields
                      
        _ = vector_store.add_documents(documents=docs_splits)
        
        state["overall_status"] = ProcessingStatus.VECTORIZED
        
        return state
    
    except Exception as e:
        logger.error(f"Error storing embeddings for documents {e}")
        state["overall_status"] = ProcessingStatus.ERROR