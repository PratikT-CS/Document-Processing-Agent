import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ..config.settings import Config
from .multi_file_state import MultiFileDocumentState, ProcessingStatus
import json
import gradio as gr
from .vector_store import vector_store
import fitz
from PIL import Image
import io
from langchain.agents import Tool
import boto3
import uuid

logger = logging.getLogger(__name__)

def extract_image(inputs: dict):
    """
    Extract images from bounding box info using bounding box details and store back extracted image to S3.
    
    Args:
    inputs(dict):{
        "file_path": str - path of uploaded file
        "page": int - page number where image is located
        "boundingBox": {
            "top": float
            "left": float
            "width": float
            "height": float
        } - bounding box details of image location in the page
    }
    Returns:
    outputs(list):{
        s3_url: str - s3uri of uploaded image
    }
    """
    try:
        if isinstance(inputs, str):
            try:
                inputs = inputs.replace('\\', '\\\\')
                inputs = json.loads(inputs)
            except Exception as e:
                logger.info(f"error: Error: Failed to parse inputs as JSON: {e}")
                return {"error": f"Something went wrong. Please try after sometime!"}
            
        file_path = inputs["file_path"]
        page_num = inputs["page"] - 1
        bbox = inputs["boundingBox"]
    
        if not all(key in inputs.keys() for key in ["file_path", "page", "boundingBox"]):
            logger.info(f"error: Error: Missing parameters in inputs")
            return {"error": f"Something went wrong. Please try after sometime!"}

        doc = fitz.open(file_path)
        page = doc.load_page(page_num)
        rect = page.rect
        page_width, page_height = rect.width, rect.height

        x0 = bbox["left"] * page_width
        y0 = bbox["top"] * page_height
        x1 = x0 + bbox["width"] * page_width
        y1 = y0 + bbox["height"] * page_height

        # Render page as image
        pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
        # img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        x0, y0, x1, y1 = [coord * (150/72) for coord in (x0, y0, x1, y1)]
        cropped_img = img.crop((x0, y0, x1, y1))
        
        buffer = io.BytesIO()
        cropped_img.save(buffer, format='PNG')
        buffer.seek(0)
        img_bytes = buffer.getvalue()
        
        doc.close()
        
        s3 = boto3.client('s3')
        bucket = "doc-processing-agent-k"
        img_key = f"cropped_imgs/{str(uuid.uuid4())}.png" 
        
        s3.put_object(Bucket="doc-processing-agent-k", Key=img_key, Body=img_bytes, ContentType="image/png")
        
        s3_url = f"https://{bucket}.s3.amazonaws.com/{img_key}"
        
        return {
            "s3Url": s3_url
        }        

    except Exception as e:
        logger.info(f"Error: Something went wrong during extracting image {e}")
        return {"error": "Please try after sometime!"}

def compare_signatures(s3_urls):
    """Compare two signature images referenced by S3 URLs using an LLM prompt.

    This function validates two provided S3 URL strings, constructs a comparison
    prompt asking the model to analyze visual similarities/differences between
    the signatures, and requests a JSON-only response containing similarity and
    confidence scores.

    Args:
        s3_urls (Dict): A dictionary containing exactly two S3 URL strings
            pointing to signature images, e.g., {"s3_url_1": "https://bucket.s3.amazonaws.com/a.png", "s3_url_2": "https://bucket.s3.amazonaws.com/b.png"}.

    Returns:
        Dict[str, Any] | str:
            - On success: a dictionary with key "comparison_result" whose value
              is the raw JSON string returned by the model (expected keys:
              "similarity_score", "confidence_score").
            - On failure: a string error message describing what went wrong.

    Raises:
        ValueError: If the input list is missing or does not contain exactly two
            URLs.
    """
    try:
        try:
            if isinstance(s3_urls, str):
                s3_urls = json.loads(s3_urls)
        except Exception as e:
            logger.info(f"error: Error: Failed to parse inputs as JSON: {e}")
            return {"error": f"Something went wrong. Please try after sometime!"}
        
        if not s3_urls or not isinstance(s3_urls, dict):
            logger.info("Invalid Input. Please provide exactly two S3 URLs.")
            raise ValueError("Invalid input. Please provide exactly two S3 URLs in a dictionary form.")
        
        llm = init_chat_model(model=Config.QnA_MODEL_NAME)
        
        prompt = f"""
        Compare the two signatures from the following image URLs and describe their similarities and differences: {s3_urls['s3_url_1']} and {s3_urls['s3_url_2']} 
        
        Give me matching score and confidence score for the result.
        
        Provide only final response in JSON format with keys 'similarity_score' and 'confidence_score'. Do not include any other text in your response.
        
        EXAMPLE:
        {{
            "similarity_score": "85%",
            "confidence_score": "98%"
        }}
        """
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        
        print(f"RESPONSE: \n{response.content}")
        
        return {"comparison_result": response.content}
        
    except Exception as e:
        logger.info(f"Error: Something went wrong during extracting image {e}")
        return f"Please try after sometime! \nError: {str(e)}"

tool_extract_image = Tool(
    name="extract_image",
    description="To extract image for provided info if visual output is required. It takes file_path, page and boundingBox information as input.",
    func=extract_image,
)

tool_signature_comparison = Tool(
    name="compare_signature",
    description="To compare two signatures from S3 URLs and determine their similarity. It takes dictionary with keys 's3_url_1' and 's3_url_2' and keys being string of s3 urls.",
    func=compare_signatures,
)

tools = [tool_extract_image, tool_signature_comparison]

class MultiFileQAAgent:
    """Question answering agent for multiple documents."""
    
    def __init__(self):

        self.llm = init_chat_model(Config.QnA_MODEL_NAME)
        
        self.multi_doc_qa_prompt = PromptTemplate(
            input_variables=["question", "relevant_context", "collection_summary", "file_list", "combined_text", "extracted_structured_data"],
            template="""
            You are answering questions about a collection of {num_files} documents. Use the provided context to give clear and concise answers.
            
            Document Collection:
            {file_list}
            
            Collection Summary:
            {collection_summary}
            
            Structured Data Extracted From Documents:
            {extracted_structured_data}
            
            Relevant Text from Documents:
            {relevant_context}
            
            User Question: {question}
            
            Instructions:
            1. Answer based on the provided context from the documents
            2. When referencing information, mention which specific document(s) it comes from
            3. If the question involves comparing documents, clearly contrast the different sources
            4. If information is missing, specify which documents were checked
            5. Provide a clear and consice answer that leverages the full document collection
            6. Use specific details and quotes when available
            7. Do not include full file path as file name only include file name in the output 
            
            Answer:
            """
        )

        self.visual_op_decider_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            You are an inteliigent assistant. Your task is to decide whether the question asked by user requires any visual output or not?

            You are replying in json format only strictly in the following form:
            {{
                "visual_output_needed": boolean
            }}

            Examples:
            1. question: Provide me details of owner from the documents.
               your answer:
                {{
                    "visual_output_needed": false
                }}
            2. question: Show me the image of the owner's signature from the documents.
               your answer:
                {{
                    "visual_output_needed": true
                }}
            3. question: Provide me driver's license number from the documents.
               your answer:
                {{
                    "visual_output_needed": true
                }} 

            NOTE: only reply in pure json object nothing else in your reply. No bacticks, no punctuation, no markdown, nothing like ```json ...``` also.

            User's Question:
            {question} 
            """
        )

        self.answer_visual_question_prompt = PromptTemplate(
            input_variables=["question", "file_list", "num_files", "combined_text", "extracted_data", "relevant_context"],
            template="""
                You are answering user's question about a collection of {num_files} documents for those questions that needs textual anwer with visual anwer as well.

                You are given with the following information:
                Document Collection: 
                {file_list}
                
                Relevant text from Documents:
                {relevant_context}
                
                Extracted data with information needed to find it in the document like bouningBox, page, file_path...
                {extracted_data}

                User's Question: {question}

                Now you have to generate both textual and visual answer for this question.
                You provide answer in following json format:
                {{
                    "text_answer": string [Answer to the question],
                    "visual_answer": List[Dict] [list of dictionaries containing visual answer from the extracted data information with 'boundingBox', 'page' and 'file_path' given to you above]
                }}
                
                NOTE: only reply in pure json object nothing else in your reply. No bacticks, no punctuation, no markdown, nothing like ```json ...``` also. And make sure file_path must be same in your response as you are provided with in extracted data information.

            ALSO MAKE SURE IN ANY CASE YOU HAVE TO PROVIDE VISUAL ANSWER AS WELL AND PROVIDE ANSWER OF VISUAL ANSWER MUST BE FROM THE EXTRACTED DATA INFORMATION GIVEN ABOVE.
            Also make sure that you should use bounding box coordinates to show the visual answer.
            """
        )
        
        self.generalized_prompt = PromptTemplate(
            input_variables=["file_list", "collection_summary", "extracted_structured_data", "extracted_data", "relevant_context", "question"],
            template="""
                You are an intelligent assistant who is answering user's questions and queries about a collection of documents. Your task is to answer the user's question in a clear and comprehensive way from the provided context from the documents.
                
                The context includes relevant text from the documents for the user's question, structured key-value pairs extracted from the documents, and information of extracted data like file_path, boundingBox, and page from the documents.
                
                You answer the user's question by executing tasks in the following order:
                1. Determine whether the user's question requires any visual output.
                2. If visual output is required, use the image extraction tool by providing file_path, page, and boundingBox information from the context. Return the text answer along with S3 URLs of images obtained from the tool.
                3. If the user wants to compare signatures across documents:
                   - First, identify the two relevant signatures from the provided extracted data information (use file_path, page, and boundingBox for each).
                   - Use the image extraction tool to extract both signatures and collect their S3 URLs.
                   - Then use the signature comparison tool by passing the two S3 URLs in A LIST to compare the signatures.
                   - EXAMPLE: Call comapare signnature comparison tool with args like the following:
                    [{{
                        "s3_url_1": "https://bucket.s3.amazon.com/signature1.png", 
                        "s3_url_2": "https://bucket.s3.amazon.com/signature2.png"  
                    }}]
                   - Finally, answer the user with a concise message that includes the comparison outcome (similarity_score and confidence_score), referencing which signature came from which document.
                4. If visual output is not required and it's not a signature comparison request, answer the question using only the provided document context.
                
                Document Collection:
                {file_list}
                
                Collection Summary:
                {collection_summary}
                
                Structured Data Extracted From Documents:
                {extracted_structured_data}
                
                Extracted data with information needed to find it in the document like boundingBox, page, file_path...
                {extracted_data}
                
                Relevant Text from Documents:
                {relevant_context}
            
                Strictly follow the following rules:
                1. Answer based on the provided context from the documents.
                2. When referencing information, mention which specific document(s) it comes from.
                3. If the question involves comparing documents, clearly contrast the different sources.
                4. If information is missing, specify which documents were checked.
                5. Provide a clear and comprehensive answer that leverages the full document collection.
                6. Use specific details and quotes when available.
                7. Use tools only when required by the user's question (e.g., visual output or signature comparison).
                8. The image extraction tool requires file_path, page, and boundingBox information from the context.
                9. Always mention which extracted image belongs to which document in your response.
                10. For signature comparison, always extract signatures first to get S3 URLs, then run the signature comparison tool with those two URLs, and include the resulting similarity_score and confidence_score in the final answer.
                11. And you must call signature comparison tool in the given format: [{{
                    "s3_url_1": "https://bucket.s3.amazon.com/signature1.png",
                    "s3_url_2": "https://bucket.s3.amazon.com/signature2.png"
                }}]
                
                User Question: {question}
            """
        )
    
    def answer_multi_document_question(self, question: str, state: MultiFileDocumentState) -> str:
        """Generate answer using context from all documents"""
        try:
            retrieved_docs_text = vector_store.similarity_search(question, filter={"type": "text"}, k=3)
            
            relevant_context = f"===\n"
            for retrieved_doc in retrieved_docs_text:
                relevant_context += f"From {retrieved_doc.metadata['source']}:\n {retrieved_doc.page_content} \n\n"
            relevant_context += "==="
            
            #Prepare file list
            files = state["files"]
            file_names = [f"- {files[file_id].file_name} ({files[file_id].file_type.upper()})" 
                        for file_id in state["file_upload_order"] 
                        if files[file_id].processing_status == ProcessingStatus.OCR_COMPLETE]
            file_list = "\n".join(file_names)
            
            structured_extracted_data = "===\n"
            for file_id, file_info in files.items():
                structured_extracted_data += f"From {file_info.file_name}: \n {file_info.extracted_data_structured}\n\n"
            structured_extracted_data += "==="
            
            # retrieved_docs_visual = vector_store.similarity_search(question, filter={"type": "key-value"}, k=6)
                
            extracted_data = []
            
            # for retrieved_doc in retrieved_docs_visual:
            #     extracted_data_obj = {'data_with_bounding_box': {}}
                
            #     extracted_data_obj['data_with_bounding_box'].update({
            #         f"{retrieved_doc.metadata['key']}": {
            #             "boundingBox": json.loads(retrieved_doc.metadata["bounding_box"]),
            #             "page": retrieved_doc.metadata["page"],
            #             "file_path": retrieved_doc.metadata["source"]
            #         }}
            #     )
            #     extracted_data.append(extracted_data_obj)
            
            for file_id, file_info in files.items():
                extracted_data.extend(file_info.extracted_data)
            
            prompt = self.generalized_prompt.format(
                question=question,
                file_list=file_list,
                relevant_context=relevant_context,
                extracted_data=extracted_data,
                extracted_structured_data=structured_extracted_data,
                collection_summary=state.get("combined_summary", "No summary available.")[:300]
            )
            
            if len(state["messages"]) == 0:
                state["messages"].append(HumanMessage(content=prompt))
                
            if len(state["messages"]) > 1:
                last_msg = state["messages"][-1]
                # print(f"LAST MESSAGE: {last_msg}")
                if hasattr(last_msg, "response_metadata"):
                    if "finish_reason" in last_msg.response_metadata:
                        if last_msg.response_metadata["finish_reason"].lower() == "stop":
                            state["messages"].append(HumanMessage(content=prompt))

            messages = state["messages"]
            
            response = self.llm.bind_tools(tools).invoke(messages)
            
            if "chat_history" not in state:
                state["chat_history"] = []
            
            if hasattr(response, "additional_kwargs"):
                if response.additional_kwargs == {}:
                    state["chat_history"].append({
                        "role": "user",
                        "content": question
                    })
                    state["chat_history"].append({
                        "role": "assistant",
                        "content": response.content.strip()
                    })

            # format chat history to reduce content size
            for message in messages:
                if isinstance(message, HumanMessage):
                    user_question = [line.strip() for line in message.content.strip().splitlines() if line.strip().startswith("User Question:")][0]
                    message.content = user_question
                    
            state["messages"] = messages
            state["messages"].append(response)
            
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error answering multi-document question: {str(e)}")
            if "chat_history" not in state:
                state["chat_history"] = []

            state["chat_history"].append({
                "role": "user",
                "content": question
            })

            state["chat_history"].append({
                "role": "assistant",
                "content": f"I apologize, but I encountered an error while processing your question about the document collection: {str(e)}"
            })
            
            if "messages" not in state:
                state["messages"] = []

            state["messages"].append({
                "role": "user",
                "content": question
            })

            state["messages"].append({
                "role": "assistant",
                "content": f"I apologize, but I encountered an error while processing your question about the document collection: {str(e)}"
            })
            return f"I apologize, but I encountered an error while processing your question about the document collection: {str(e)}"

def process_multi_document_question(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """
    LangGraph node: Process user question across all documents
    """
    try:
        # Get question from state
        current_query = state.get("current_query")
        if not current_query:
            state["response"] = "No question provided."
            return state
        
        # Check if documents are ready
        if state.get("overall_status") != ProcessingStatus.VECTORIZED:
            state["response"] = "Documents are not ready for questions yet. Please wait for processing to complete."
            return state
        
        # Initialize multi-file QA agent
        qa_agent = MultiFileQAAgent()
        
        # Generate answer
        logger.info(f"Processing multi-document question: {current_query[:100]}...")
        answer = qa_agent.answer_multi_document_question(current_query, state)
        
        logger.info("Question processed successfully")
        
        return state
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        state["response"] = f"I apologize, but I encountered an error while processing your question: {str(e)}"
        if "chat_history" not in state:
            state["chat_history"] = []
        state["chat_history"].append({"role": "user", "content": current_query})
        state["chat_history"].append({"role": "assistant", "content": f"I apologize, but I encountered an error while processing your question: {str(e)}"})
        return state
    
def format_response_for_gradio(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """Format response for Gradio UI."""
    try:
        last_response = state["messages"][-1]
        chat_history = state["chat_history"]
        last_message = chat_history.pop()
        
        # print(f"Response: {last_response}")
        # print(f"Last Message: {last_message}")
        
        if not last_message["content"] or last_response.content == "":
            raise (f"No response content or last message content ia available")

        prompt_for_format_response = PromptTemplate(
            input_variables=["question", "llm_responnse"],
            template="""
            You are an intelligent assistant who helps format reponse from LLM into json format for UI.
            You are provided with the user question and response from LLM. You need to extract textual answer and valid s3 uris from the LLM response. You can ignore any other urls other than valid s3 uris in the LLM response. Also note that s3 uris are present only if user's question requires any visual output.
            
            You need to format the response in following json format:
            {{
                "text_answer": string [textual anwer for user question],
                "s3_uris": List[Dict] [list of valid s3 URIs in the LLM reponse for user's question with their labels to show in the UI.]
            }}
            
            Example response:
            {{
                "text_answer": "Here are the signatures from the documents.",
                "s3_uris": [
                    {{"label": "Image 1", "s3_uri": "https://example.com/image1.jpg"}},
                    {{"label": "Image 2", "s3_uri": "https://example.com/image2.jpg"}}
                ]
            }}

            User's question: {question}
            
            LLM response: {llm_response}
            
            NOTE: 
            - Only reply in pure json object's string value, nothing else in your reply. No bacticks, no punctuation, no markdown, nothing like maerkdown json also. And do not include full file path as file name if present, only include file name in the text_answer field.
            - Do not completely modify original LLM response, only extract valid s3 uris and text answer for user's question.
            """
        )
        
        llm = init_chat_model(Config.QnA_MODEL_NAME)
        
        response = llm.invoke(prompt_for_format_response.format(question=state["current_query"], llm_response=last_response.content.strip()))
        
        if response.content.startswith("```json"):
            formatted_response = response.content.replace('```json', '').replace('```', '')
            formatted_response = json.loads(formatted_response.strip())
        elif response.content.startswith("{"):
            formatted_response = json.loads(response.content.strip())
        else: 
            raise ValueError("Invalid JSON format in response.")
        
        chat_history.append({
            "role": "assistant",
            "content": formatted_response["text_answer"]
        })
        
        for s3_uri in formatted_response["s3_uris"]:
            chat_history.append({
                "role": "assistant",
                "content": f"Image: {s3_uri['label']}"
            })
            chat_history.append({
                "role": "assistant",
                "content": gr.Image(
                    value=s3_uri["s3_uri"],
                    label=s3_uri["label"],
                    show_label=True
                )
            })
        
        logger.info(f"Response formatted and added to chat history.")
        return state
        
    except Exception as e:
        logger.error(f"Error formatting response for Gradio UI: {str(e)}")
        chat_history.append({
            "role": "assistant",
            "content": f"I apologize, but I encountered an error while formatting the response for Gradio UI: {str(e)}"
        })
        return state    
         