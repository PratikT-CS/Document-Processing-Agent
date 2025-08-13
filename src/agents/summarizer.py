import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from ..config.settings import Config
from .graph_state import DocumentProcessingState

logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """Handles document summarization and question generation"""
    
    def __init__(self):
        # self.llm = ChatOpenAI(
        #     model=Config.MODEL_NAME,
        #     temperature=Config.TEMPERATURE,
        #     max_tokens=Config.MAX_TOKENS,
        #     openai_api_key=Config.OPENAI_API_KEY
        # )

        self.llm = init_chat_model(Config.MODEL_NAME)
        
        self.summary_prompt = PromptTemplate(
            input_variables=["document_text", "metadata"],
            template="""
            Analyze the following document and provide a comprehensive summary.
            
            Document Metadata:
            - Word Count: {metadata[word_count]}
            - Estimated Reading Time: {metadata[estimated_reading_time]} minutes
            
            Document Content:
            {document_text}
            
            Please provide:
            1. A clear, concise summary (2-3 paragraphs) that captures the main points and key information
            2. The document type/category (e.g., research paper, business report, manual, etc.)
            3. Key topics or themes covered
            
            Format your response as:
            **Summary:**
            [Your summary here]
            
            **Document Type:**
            [Document type]
            
            **Key Topics:**
            [List of key topics]
            """
        )
        
        self.questions_prompt = PromptTemplate(
            input_variables=["document_text", "summary"],
            template="""
            Based on the following document summary and content, generate 5 relevant questions that users might want to ask about this document.
            
            Summary:
            {summary}
            
            Document Content (first 2000 characters):
            {document_text}
            
            Generate questions that:
            1. Cover different aspects of the document
            2. Are specific and answerable from the content
            3. Range from basic factual questions to more analytical ones
            4. Would be genuinely useful for someone trying to understand the document
            
            Format as a simple list:
            1. Question 1
            2. Question 2
            3. Question 3
            4. Question 4
            5. Question 5
            """
        )
    
    def generate_summary(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate document summary"""
        try:
            # Truncate text if too long (keep first portion for summary)
            text_for_summary = text[:8000] if len(text) > 8000 else text
            
            prompt = self.summary_prompt.format(
                document_text=text_for_summary,
                metadata=metadata
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def generate_questions(self, text: str, summary: str) -> List[str]:
        """Generate suggested questions about the document"""
        try:
            # Use first 2000 characters for question generation
            text_sample = text[:2000] if len(text) > 2000 else text
            
            prompt = self.questions_prompt.format(
                document_text=text_sample,
                summary=summary
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse questions from response
            questions = self._parse_questions(response.content)
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return [
                "What is the main topic of this document?",
                "What are the key points discussed?",
                "Who is the intended audience?",
                "What conclusions or recommendations are made?",
                "Are there any important dates or numbers mentioned?"
            ]
    
    def _parse_questions(self, response_text: str) -> List[str]:
        """Parse questions from LLM response"""
        questions = []
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered questions
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering and clean up
                question = line.split('.', 1)[-1].strip()
                question = question.lstrip('- •').strip()
                if question and question.endswith('?'):
                    questions.append(question)
        
        # If parsing failed, try to extract any questions
        if not questions:
            import re
            questions = re.findall(r'[A-Z][^?]*\?', response_text)
        
        return questions[:5]  # Return max 5 questions

def generate_document_summary_and_questions(state: DocumentProcessingState) -> DocumentProcessingState:
    """
    LangGraph node: Generate document summary and suggested questions
    """
    try:
        # Update processing status
        state["processing_status"] = "processing"
        state["current_step"] = "summarize"
        state["processing_progress"] = 85
        
        # Get processed text and metadata
        processed_text = state.get("processed_text", "")
        metadata = state.get("document_metadata", {})
        
        if not processed_text.strip():
            raise Exception("No processed text available for summarization")
        
        # Initialize summarizer
        summarizer = DocumentSummarizer()
        
        # Generate summary
        logger.info("Generating document summary...")
        summary = summarizer.generate_summary(processed_text, metadata)
        
        # Generate questions
        logger.info("Generating suggested questions...")
        questions = summarizer.generate_questions(processed_text, summary)
        
        # Update state
        state["document_summary"] = summary
        state["suggested_questions"] = questions
        state["processing_status"] = "summarized"
        state["current_step"] = "chat"
        state["processing_progress"] = 100
        
        logger.info(f"Summary and questions generated successfully. {len(questions)} questions created.")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        state["processing_status"] = "error"
        state["error_message"] = f"Summarization failed: {str(e)}"
        return state