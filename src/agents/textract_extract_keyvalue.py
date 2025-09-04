import fitz
import boto3
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

DEFAULT_BBOX = [
    {
        "BoundingBox": {
            "Left": float("inf"),
            "Top": float("inf"),
            "Width": 0,
            "Height": 0,
        },
        "Page": 1,
    }
]

def merge_bounding_boxes(bboxes):
    if not bboxes:
        return None  # no input
    bboxes_to_merge = [
        geometry[0]["BoundingBox"] for geometry in bboxes if len(geometry) > 0
    ]
    page = bboxes[0][0]["Page"]

    left = min(b["Left"] for b in bboxes_to_merge)
    top = min(b["Top"] for b in bboxes_to_merge)
    right = max(b["Left"] + b["Width"] for b in bboxes_to_merge)
    bottom = max(b["Top"] + b["Height"] for b in bboxes_to_merge)

    return {
        "BoundingBox": {
            "Left": left,
            "Top": top,
            "Width": right - left,
            "Height": bottom - top,
        },
        "Page": page,
    }

def get_kv_map(response):
    # Get the text blocks
    blocks = response["Blocks"]
    # get key and value maps
    key_map = {}
    value_map = {}
    block_map = {}
    for block in blocks:
        block_id = block["Id"]
        block_map[block_id] = block
        if block != None:
            if block["BlockType"] == "KEY_VALUE_SET":
                if "KEY" in block["EntityTypes"]:
                    key_map[block_id] = block
                else:
                    value_map[block_id] = block

    return key_map, value_map, block_map

def get_kv_relationship(key_map, value_map, block_map, page_num):
    kvs = defaultdict(list)
    for block_id, key_block in key_map.items():
        value_block = find_value_block(key_block, value_map)
        key, bboxes_key = get_text(key_block, block_map, page_num)
        val, bboxes_val = get_text(value_block, block_map, page_num)
        key_bbox = merge_bounding_boxes(bboxes_key)
        val_bbox = merge_bounding_boxes(bboxes_val)
        kv_set_bbox = merge_bounding_boxes([[key_bbox], [val_bbox]])
        kv_set_bbox = dict((key.lower(), value) for (key, value) in kv_set_bbox.items())
        kvs[key].extend([val, kv_set_bbox])
    return kvs

def find_value_block(key_block, value_map):
    for relationship in key_block["Relationships"]:
        if relationship["Type"] == "VALUE":
            for value_id in relationship["Ids"]:
                value_block = value_map[value_id]
    return value_block

def get_text(result, blocks_map, page_num):
    text = ""
    bboxes = []
    if "Relationships" in result:
        for relationship in result["Relationships"]:
            if relationship["Type"] == "CHILD":
                for child_id in relationship["Ids"]:
                    word = blocks_map[child_id]
                    bboxes.append(
                        [
                            {
                                "BoundingBox": word["Geometry"]["BoundingBox"],
                                "Page": page_num,
                            }
                        ]
                    )
                    if word["BlockType"] == "LINE":
                        print("LINE")
                        text += word["Text"]
                    if word["BlockType"] == "WORD":
                        text += word["Text"] + " "
                    if word["BlockType"] == "SELECTION_ELEMENT":
                        if word["SelectionStatus"] == "SELECTED":
                            text += "✔️"
                        else:
                            text += "❌"
    else:
        return "", [DEFAULT_BBOX]

    return text, bboxes

def print_kvs(kvs):
    for key, value in kvs.items():
        print(key, ":", value)

def extract_kvs(file_path:str):
    extracted_data = []
    try:
        textract_client = boto3.client("textract")
        doc = fitz.open(filename=file_path, filetype="pdf")
        logger.info(f"Extrating key-value pairs from {file_path} using textract.")
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
            img_bytes = pix.tobytes()
            
            if len(img_bytes) > 5 * 1024 * 1024:  # 5MB limit
                logger.info(f"Warning: Page {page_num + 1} image is too large for Textract. Skipping.")
                continue
            
            analyze_doc_response = textract_client.analyze_document(Document={'Bytes': img_bytes}, FeatureTypes=['FORMS'], )
            
            key_map, val_map, block_map = get_kv_map(analyze_doc_response)
            page_kvs = get_kv_relationship(key_map, val_map, block_map, page_num+1)
            
            for key, value in page_kvs.items():
                value_obj = {}
                value[1]['boundingbox'] = dict((key.lower(), value) for (key, value) in value[1]['boundingbox'].items())
                value_obj["geometry"] = [
                    {
                        "boundingBox": value[1]['boundingbox'],
                        "page": value[1]['page'],
                    }
                ]
                value_obj["value"] = value[0]
                value_obj["type"]= "string"
                extracted_data.append({
                    key: value_obj
                })
                
        logger.info(f"Extracted KVs: \n\n{extracted_data}\n\n")
        logger.info(f"Key-Value pairs extracted successfully.")
        return extracted_data
    except Exception as e:
        logger.info(f"An error occured during extracting key-value pairs from {file_path}")
        logger.info(f"Error: {e}")
        return defaultdict(list)
    