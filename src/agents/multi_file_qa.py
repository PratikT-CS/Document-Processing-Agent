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

logger = logging.getLogger(__name__)

class MultiFileQAAgent:
    """Question answering agent for multiple documents"""
    
    def __init__(self):
        # self.llm = ChatOpenAI(
        #     model=Config.MODEL_NAME,
        #     temperature=Config.TEMPERATURE,
        #     max_tokens=Config.MAX_TOKENS,
        #     openai_api_key=Config.OPENAI_API_KEY
        # )

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
    
    def answer_multi_document_question(self, question: str, state: MultiFileDocumentState) -> str:
        """Generate answer using context from all documents"""
        try:
            prompt = self.visual_op_decider_prompt.format(
                question=question
            )

            response = self.llm.invoke([HumanMessage(content=prompt)])
            logger.info(f"Decider response: {response.content.strip()}")
            response = json.loads(response.content.strip())


            if not response["visual_output_needed"]:
                retrieved_docs_text = vector_store.similarity_search(question, filter={"type": "text"}, k=7)
                
                relevant_context = f"===\n"
                for retrieved_doc in retrieved_docs_text:
                    relevant_context += f"From {retrieved_doc.metadata['source']}:\n {retrieved_doc.page_content} \n\n"
                relevant_context += "==="
                
                # Prepare file list
                files = state["files"]
                file_names = [f"- {files[file_id].file_name} ({files[file_id].file_type.upper()})" 
                            for file_id in state["file_upload_order"] 
                            if files[file_id].processing_status == ProcessingStatus.OCR_COMPLETE]
                file_list = "\n".join(file_names)
                
                structured_extracted_data = "===\n"
                for file_id, file_info in files.items():
                    structured_extracted_data += f"From {file_info.file_name}: \n {file_info.extracted_data_structured}\n\n"
                structured_extracted_data += "==="
                
                # Generate answer
                prompt = self.multi_doc_qa_prompt.format(
                    question=question,
                    relevant_context=relevant_context,
                    collection_summary=state.get("combined_summary", "No summary available."),
                    file_list=file_list,
                    num_files=len(file_names),
                    combined_text=state.get("combined_text"),
                    extracted_structured_data=structured_extracted_data
                )
                
                response = self.llm.invoke([HumanMessage(content=prompt)])

                 # Update state
                state["response"] = response.content.strip()
                # state["relevant_chunks"] = relevant_chunks
                
                # Add to chat history
                if "chat_history" not in state:
                    state["chat_history"] = []
                
                state["chat_history"].append({
                    "role": "user",
                    "content": question
                })

                state["chat_history"].append({
                    "role": "assistant",
                    "content": response.content.strip()
                })

                return response.content.strip()
            
            else:
                logger.info(f"Visual info needed {question}")
                # Prepare file list
                files = state["files"]
                file_names = [f"- {files[file_id].file_name} ({files[file_id].file_type.upper()})" 
                            for file_id in state["file_upload_order"] 
                            if files[file_id].processing_status == ProcessingStatus.OCR_COMPLETE]
                file_list = "\n".join(file_names)
                
                retrieved_docs_text = vector_store.similarity_search(question, filter={"type": "text"}, k=3)
                
                relevant_context = f"===\n"
                for retrieved_doc in retrieved_docs_text:
                    relevant_context += f"From {retrieved_doc.metadata['source']}:\n {retrieved_doc.page_content} \n\n"
                relevant_context += "==="
                
                retrieved_docs_visual = vector_store.similarity_search(question, filter={"type": "key-value"}, k=8)
                
                extracted_data = []
                
                for retrieved_doc in retrieved_docs_visual:
                    extracted_data_obj = {'data_with_bounding_box': {}}
                    
                    extracted_data_obj['data_with_bounding_box'].update({
                        f"{retrieved_doc.metadata['key']}": {
                            "boundingBox": json.loads(retrieved_doc.metadata["bounding_box"]),
                            "page": retrieved_doc.metadata["page"],
                            "file_path": retrieved_doc.metadata["source"]
                        }}
                    )
                    extracted_data.append(extracted_data_obj)
                
                # for file_id, file_info in files.items():
                #     extracted_data_obj = {file_info.file_name: {}}

                #     extracted_data_obj[file_info.file_name].update({"data_with_bounding_box": file_info.extracted_data})
                #     extracted_data.append(extracted_data_obj)

                # Generate answer
                prompt = self.answer_visual_question_prompt.format(
                    question=question,
                    file_list=file_list,
                    num_files=len(file_names),
                    combined_text=state.get("combined_text"),
                    extracted_data=extracted_data,
                    relevant_context=relevant_context
                )
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                
                info = json.loads(response.content.strip())
                print(info)
                info_visual = info["visual_answer"]
                images = extract_image(info_visual)

                if "chat_history" not in state:
                    state["chat_history"] = []

                state["chat_history"].append({
                    "role": "user",
                    "content": question
                })

                state["chat_history"].append({
                    "role": "assistant",
                    "content": info["text_answer"]
                })

                for image in images:
                    state["chat_history"].append({
                        "role": "assistant",
                        "content": gr.Image(value=image)
                    })

                return info["text_answer"]
            
        except Exception as e:
            logger.error(f"Error answering multi-document question: {str(e)}")
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
    
def extract_image(items):
    cropped_images = []

    for item in items:
        file_path = item["file_path"]
        page_num = item["page"] - 1
        bbox = item["boundingBox"]

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
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        x0, y0, x1, y1 = [coord * (150/72) for coord in (x0, y0, x1, y1)]

        # Crop image
        cropped_img = img.crop((x0, y0, x1, y1))
        cropped_images.append(cropped_img)

        doc.close()

    return cropped_images