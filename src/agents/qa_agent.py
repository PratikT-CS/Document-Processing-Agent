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
from .graph_state import DocumentProcessingState

logger = logging.getLogger(__name__)

class DocumentQAAgent:
    """Handles question answering for documents"""
    
    def __init__(self):
        # self.llm = ChatOpenAI(
        #     model=Config.MODEL_NAME,
        #     temperature=Config.TEMPERATURE,
        #     max_tokens=Config.MAX_TOKENS,
        #     openai_api_key=Config.OPENAI_API_KEY
        # )

        self.llm = init_chat_model(Config.MODEL_NAME)
        
        self.qa_prompt = PromptTemplate(
            input_variables=["question", "context", "document_summary", "processed_text"],
            template="""
            You are an AI assistant helping users understand a document. Answer the question based on the provided context and document summary.
            
            Document Summary:
            {document_summary}
            
            Raw Text of the Documuent:
            {processed_text}
            
            User Question: {question}
            
            Instructions:
            1. Answer the question directly and concisely based on the provided context
            2. If the information is not in the context, say so clearly
            3. Use specific details from the document when possible
            4. If the question asks for something not covered in the document, explain what information is available instead
            5. Keep your response focused and relevant to the question
            
            Answer:
            """
        )
        
        # Initialize TF-IDF vectorizer for simple retrieval
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.chunk_vectors = None
        self.chunks = None
    
    def prepare_retrieval_index(self, chunks: List[str]) -> None:
        """Prepare retrieval index from document chunks"""
        try:
            if not chunks:
                logger.warning("No chunks provided for indexing")
                return
            
            self.chunks = chunks
            self.chunk_vectors = self.vectorizer.fit_transform(chunks)
            logger.info(f"Prepared retrieval index with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error preparing retrieval index: {str(e)}")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant chunks for the query"""
        try:
            if not self.chunks or self.chunk_vectors is None:
                logger.warning("No retrieval index available")
                return []
            
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
            
            # Get top k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            relevant_chunks = [self.chunks[i] for i in top_indices if similarities[i] > 0.1]
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for query")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []
    
    def answer_question(self, question: str, document_summary: str, chunks: List[str], processed_text) -> str:
        """Generate answer to user question"""
        try:
            # Retrieve relevant context
            if not self.chunks:
                self.prepare_retrieval_index(chunks)
            
            relevant_chunks = self.retrieve_relevant_chunks(question)
            
            # Prepare context
            context = "\n\n".join(relevant_chunks) if relevant_chunks else "No specific context found."
            
            # Limit context length
            if len(context) > 3000:
                context = context[:3000] + "..."
            
            # Generate answer
            prompt = self.qa_prompt.format(
                question=question,
                context=context,
                document_summary=document_summary,
                processed_text=processed_text
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"

def process_user_question(state: DocumentProcessingState) -> DocumentProcessingState:
    """
    LangGraph node: Process user question and generate answer
    """
    try:
        # Get question from state
        current_query = state.get("current_query")
        if not current_query:
            state["response"] = "No question provided."
            return state
        
        # Get document data
        document_summary = state["document_summary"]
        document_chunks = state["document_chunks"]
        processed_text = state["processed_text"]
        
        if not document_summary and not document_chunks:
            state["response"] = "Document has not been processed yet. Please upload and process a document first."
            return state
        
        # Initialize QA agent
        qa_agent = DocumentQAAgent()
        
        # Generate answer
        logger.info(f"Processing question: {current_query[:100]}...")
        answer = qa_agent.answer_question(current_query, document_summary, document_chunks, processed_text)
        
        # Update state
        state["response"] = answer
        
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
        
        # Store relevant chunks for reference
        if hasattr(qa_agent, 'chunks') and qa_agent.chunks:
            state["relevant_chunks"] = qa_agent.retrieve_relevant_chunks(current_query)
        
        logger.info("Question processed successfully")
        
        return state
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        state["response"] = f"I apologize, but I encountered an error while processing your question: {str(e)}"
        return state