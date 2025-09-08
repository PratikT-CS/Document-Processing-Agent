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

def merge_bounding_boxes(bboxes):
    if not bboxes:
        return None  # no input
    
    left = min(b["left"] for b in bboxes)
    top = min(b["top"] for b in bboxes)
    right = max(b["left"] + b["width"] for b in bboxes)
    bottom = max(b["top"] + b["height"] for b in bboxes)
    
    return {
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top
    }
        
def store_embeddings(state: MultiFileDocumentState) -> MultiFileDocumentState:
    if state["overall_status"] == ProcessingStatus.VECTORIZED:
        return state
    
    try:
        files = state["files"]
        all_docs = []
        
        for file_id, file_info in files.items():
            # Process text document
            text_doc = Document(
                page_content=file_info.processed_text,
                id=str(uuid.uuid4()),
                metadata={
                    "type": "text",
                    "source": file_info.file_name,
                    "bounding_box": "{}",
                    "page": 0,
                    "key": "",
                    "value": "",
                    "type-value": ""
                }
            )
            all_docs.extend(text_splitter.split_documents([text_doc]))
            
            # Process extracted data
            flatten_data = file_info.extracted_data if len(file_info.extracted_data) > 1 else flatten_kv_items(file_info.extracted_data)
            
            # Remove vertices to reduce storage
            [value['geometry'][0].pop('vertices', None) for pair in flatten_data for (key, value) in pair.items() if not isinstance(value, list) and "geometry" in value.keys() and value["geometry"]]
            
            extracted_fields = {}
            
            for item in flatten_data:
                for key, value in item.items():
                    if isinstance(value, list) or "geometry" not in value or not value["geometry"]:
                        continue
                        
                    extracted_fields[key] = value["value"]
                    bbox = merge_bounding_boxes([g["boundingBox"] for g in value["geometry"]])
                    
                    all_docs.append(Document(
                        page_content=f"{key} is {value['value'] if not isinstance(value['value'], bool) else 'present' if value['value'] else 'not present'}",
                        id=str(uuid.uuid4()),
                        metadata={
                            "type": "key-value",
                            "source": file_info.file_name,
                            "bounding_box": json.dumps(bbox),
                            "page": value["geometry"][0]["page"],
                            "key": str(key),
                            "value": value["value"],
                            "type-value": value["type"]
                        }
                    ))
            
            file_info.extracted_data = flatten_data
            file_info.extracted_data_structured = extracted_fields
        
        vector_store.add_documents(documents=all_docs)
        
        state["overall_status"] = ProcessingStatus.VECTORIZED
        state["uploaded_file_paths"] = []
        return state
        
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        state["overall_status"] = ProcessingStatus.ERROR
        return state