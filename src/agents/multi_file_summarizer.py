import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from ..config.settings import Config
from .multi_file_state import MultiFileDocumentState, ProcessingStatus
import json

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS = [
    "What are the main topics covered across all documents?",
    "How do these documents relate to each other?",
    "What are the key findings from this document collection?",
    "Which document provides the most detailed information on [topic]?",
    "Are there any contradictions between the documents?",
    "What common themes appear in multiple documents?",
    "What unique insights does each document provide?",
    "What conclusions can be drawn from this collection?"
]

class MultiFileDocumentSummarizer:
    """Handles summarization of multiple documents"""
    
    def __init__(self):
        # self.llm = ChatOpenAI(
        #     model=Config.MODEL_NAME,
        #     temperature=Config.TEMPERATURE,
        #     max_tokens=Config.MAX_TOKENS,
        #     openai_api_key=Config.OPENAI_API_KEY
        # )

        self.llm = init_chat_model(Config.SUMMARIZER_MODEL_NAME)
        
        self.multi_doc_summarization_and_questions_prompt = PromptTemplate(
            input_variables=["documents_info", "combined_text_sample", "num_documents"],
            template="""
You are analyzing a collection of {num_documents} documents. Provide a comprehensive analysis and suggested questions one casn ask about the documents.

Document Collection Overview:
{documents_info}

Combined Content Sample:
{combined_text_sample}

Instructions:
    - For document summary:
    1. **Collection Summary**: Overall summary of all documents and their main themes
    2. **Individual Document Insights**: Brief summary of what each document contributes
    3. **Common Themes**: Topics or themes that appear across multiple documents
    4. **Document Relationships**: How the documents relate to each other (complementary, contrasting, etc.)
    5. **Key Findings**: Most important insights from the entire collection
- For suggested questions:
Generate exactly 4 thought-provoking questions that:
   - Focus on document relationships and comparisons
   - Explore cross-document themes and patterns
   - Combine factual and analytical aspects
   - Cover both specific details and broader insights

Please provide your final answer in pure json format that only 2 things as below:
{{
"documents_summary": "Markdown for documents summary as described in instructions.",
"suggested_questions": List(str) - "Suggested questions in a list as described in instuctions."
}}

NOTE: Do not include full file path as file name only include file name in the output and only reply in pure json object's string value, nothing else in your reply.
            """
        )
    
    def generate_combined_summary_and_suggested_questions(self, state: MultiFileDocumentState) -> str:
        """Generate summary and suggested questionms for the entire document collection"""
        try:
            files = state["files"]
            combined_text = state.get("combined_text", "")
            
            if not combined_text:
                raise Exception("No combined text available for summarization")
            
            # Prepare document information
            docs_info = []
            for file_id in state["file_upload_order"]:
                file_info = files[file_id]
                if file_info.processing_status == ProcessingStatus.OCR_COMPLETE:
                    metadata = file_info.document_metadata
                    docs_info.append(
                        f"- **{file_info.file_name}** ({file_info.file_type.upper()}): "
                        f"{metadata.get('word_count', 'Unknown')} words, "
                        f"{metadata.get('paragraph_count', 'Unknown')} paragraphs"
                    )
            
            documents_info = "\n".join(docs_info)
            combined_text_sample = combined_text
            
            # Generate summary
            prompt = self.multi_doc_summarization_and_questions_prompt.format(
                num_documents=len(docs_info),
                documents_info=documents_info,
                combined_text_sample=combined_text_sample
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            if response.content.startswith("```json"):
                response = response.content.replace('```json', '').replace('```', '')
                response = json.loads(response.strip())
            elif response.content.startswith("{"):
                response = json.loads(response.content.strip())
            else: 
                raise ValueError("Invalid JSON format in response.")
            
            return response["documents_summary"], response["suggested_questions"]
            
        except Exception as e:
            logger.error(f"Error generating combined summary: {str(e)}")
            return f"Error generating summary: {str(e)}", DEFAULT_QUESTIONS

def generate_multi_document_summary(state: MultiFileDocumentState) -> MultiFileDocumentState:
    """
    LangGraph node: Generate summary for all documents
    """
    try:
        # state["overall_status"] = ProcessingStatus.PROCESSING
        state["current_step"] = "summarize"
        
        # Check if OCR is complete
        if state["overall_status"] != ProcessingStatus.OCR_COMPLETE:
            raise Exception("OCR processing must be complete before summarization")
        
        # Initialize summarizer
        summarizer = MultiFileDocumentSummarizer()
        
        # Generate combined summary
        logger.info("Generating combined summary and questions for all documents")
        combined_summary, questions = summarizer.generate_combined_summary_and_suggested_questions(state)
        
        # Update state
        state["combined_summary"] = combined_summary
        state["suggested_questions"] = questions
        state["document_relationships"] = []
        state["overall_status"] = ProcessingStatus.SUMMARIZED
        state["processing_progress"]["overall"] = 100
        
        logger.info(f"Multi-document summarization completed. {len(questions)} questions generated.")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in multi-document summarization: {str(e)}")
        state["overall_status"] = ProcessingStatus.ERROR
        state["error_message"] = f"Summarization failed: {str(e)}"
        return state
