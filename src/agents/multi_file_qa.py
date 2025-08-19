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

        self.llm = init_chat_model(Config.MODEL_NAME)
        
        self.multi_doc_qa_prompt = PromptTemplate(
            input_variables=["question", "relevant_chunks", "collection_summary", "file_list", "combined_text"],
            template="""
            You are answering questions about a collection of {num_files} documents. Use the provided context to give comprehensive answers.
            
            Document Collection:
            {file_list}
            
            Collection Summary:
            {collection_summary}
            
            Combined Text (all documents):
            {combined_text}
            
            User Question: {question}
            
            Instructions:
            1. Answer based on the provided context from the documents
            2. When referencing information, mention which specific document(s) it comes from
            3. If the question involves comparing documents, clearly contrast the different sources
            4. If information is missing, specify which documents were checked
            5. Provide a comprehensive answer that leverages the full document collection
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
            input_variables=["question", "file_list", "num_files", "combined_text", "extracted_data"],
            template="""
                You are answering user's question about a collection of {num_files} documents for those questions that needs textual anwer with visual anwer as well.

                You are given with the following information:
                Document Collection: 
                {file_list}
                
                Combined Text (all documents):
                {combined_text}
                
                Extracted data with information needed to find it in the doocuent like bouningBox, page...
                {extracted_data}

                User's Question: {question}

                Now you have to generate both textual and visual answer for this question.
                You provide answer in following json format:
                {{
                    "text_answer": string [Answer to the question],
                    "visual_answer": List[Dict] [list of dictionaries containing visual answer from the extracted data information given to you above]
                }}
                
                NOTE: only reply in pure json object nothing else in your reply. No bacticks, no punctuation, no markdown, nothing like ```json ...``` also.

            ALSO MAKE SURE IN ANY CASE YOU HAVE TO PROVIDE VISUAL ANSWER AS WELL AND PROVIDE ANSWER OF VISUAL ANSWER MUST BE FROM THE EXTRACTED DATA INFORMATION GIVEN ABOVE.
            Also make sure that you should use bounding box coordinates to show the visual answer.
            """
        )
        
        # TF-IDF vectorizer for retrieval across all documents
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english', 
            ngram_range=(1, 2)
        )
        self.chunk_vectors = None
        self.chunks_with_metadata = None
    
    def prepare_multi_file_index(self, state: MultiFileDocumentState) -> None:
        """Prepare retrieval index from all document chunks"""
        try:
            combined_chunks = state.get("combined_chunks", [])
            if not combined_chunks:
                logger.warning("No combined chunks available for indexing")
                return
            
            # Extract chunk content for vectorization
            chunk_contents = [chunk["content"] for chunk in combined_chunks]
            self.chunks_with_metadata = combined_chunks
            
            # Create TF-IDF vectors
            self.chunk_vectors = self.vectorizer.fit_transform(chunk_contents)
            
            logger.info(f"Prepared multi-file retrieval index with {len(combined_chunks)} chunks from {len(state['files'])} documents")
            
        except Exception as e:
            logger.error(f"Error preparing multi-file index: {str(e)}")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks across all documents"""
        try:
            if not self.chunks_with_metadata or self.chunk_vectors is None:
                logger.warning("No multi-file index available")
                return []
            
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
            
            # Get top k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            relevant_chunks = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    chunk_data = self.chunks_with_metadata[idx].copy()
                    chunk_data["similarity_score"] = similarities[idx]
                    relevant_chunks.append(chunk_data)
            
            # Sort by similarity score
            relevant_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks from multiple documents")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    def answer_multi_document_question(self, question: str, state: MultiFileDocumentState) -> str:
        """Generate answer using context from all documents"""
        try:
            # Prepare index if not already done
            if not self.chunks_with_metadata:
                self.prepare_multi_file_index(state)
            
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(question, top_k=6)

            prompt = self.visual_op_decider_prompt.format(
                question=question
            )

            response = self.llm.invoke([HumanMessage(content=prompt)])
            logger.info(f"Decider response: {response.content.strip()}")
            response = json.loads(response.content.strip())


            if not response["visual_output_needed"]:
                # Prepare context from relevant chunks
                context_parts = []
                for chunk in relevant_chunks:
                    source_info = f"[From: {chunk['file_name']}]"
                    context_parts.append(f"{source_info}\n{chunk['content']}")
                
                relevant_context = "\n\n".join(context_parts) if context_parts else "No specific relevant context found."
                
                # Limit context length
                if len(relevant_context) > 4000:
                    relevant_context = relevant_context[:4000] + "\n\n[Context truncated...]"
                
                # Prepare file list
                files = state["files"]
                file_names = [f"- {files[file_id].file_name} ({files[file_id].file_type.upper()})" 
                            for file_id in state["file_upload_order"] 
                            if files[file_id].processing_status == ProcessingStatus.OCR_COMPLETE]
                file_list = "\n".join(file_names)
                
                # Generate answer
                prompt = self.multi_doc_qa_prompt.format(
                    question=question,
                    relevant_chunks=relevant_context,
                    collection_summary=state.get("combined_summary", "No summary available."),
                    file_list=file_list,
                    num_files=len(file_names),
                    combined_text=state.get("combined_text")
                )
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content.strip()
            
            else:
                logger.info(f"Visual info needed {question}")
                # Prepare file list
                files = state["files"]
                file_names = [f"- {files[file_id].file_name} ({files[file_id].file_type.upper()})" 
                            for file_id in state["file_upload_order"] 
                            if files[file_id].processing_status == ProcessingStatus.OCR_COMPLETE]
                file_list = "\n".join(file_names)
                
                extracted_data = []
                for file_id, file_info in files.items():
                    extracted_data_obj = {file_info.file_name: {}}
                    extracted_data_obj[file_info.file_name].update({"data_with_bounding_box": file_info.extracted_data})
                    extracted_data.append(extracted_data_obj)

                # Generate answer
                prompt = self.answer_visual_question_prompt.format(
                    question=question,
                    file_list=file_list,
                    num_files=len(file_names),
                    combined_text=state.get("combined_text"),
                    extracted_data=extracted_data
                )
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content.strip()
            
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
        if state.get("overall_status") != ProcessingStatus.SUMMARIZED:
            state["response"] = "Documents are not ready for questions yet. Please wait for processing to complete."
            return state
        
        # Initialize multi-file QA agent
        qa_agent = MultiFileQAAgent()
        
        # Generate answer
        logger.info(f"Processing multi-document question: {current_query[:100]}...")
        answer = qa_agent.answer_multi_document_question(current_query, state)
        
        # Get relevant chunks for reference
        relevant_chunks = qa_agent.retrieve_relevant_chunks(current_query)
        
        # Update state
        state["response"] = answer
        state["relevant_chunks"] = relevant_chunks
        
        # Add to chat history
        if "chat_history" not in state:
            state["chat_history"] = []
        
        state["chat_history"].append({
            "role": "user",
            "content": current_query
        })

        state["chat_history"].append({
            "role": "assistant",
            "content": answer
        })

        if hasattr(qa_agent, 'chunks_with_metadata') and qa_agent.chunks_with_metadata:
            state["relevant_chunks"] = qa_agent.retrieve_relevant_chunks(current_query)
        
        logger.info("Question processed successfully")
        
        return state
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        state["response"] = f"I apologize, but I encountered an error while processing your question: {str(e)}"
        return state