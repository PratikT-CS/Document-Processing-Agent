import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from ..config.settings import Config
from .multi_file_state import MultiFileDocumentState, ProcessingStatus

logger = logging.getLogger(__name__)

class MultiFileDocumentSummarizer:
    """Handles summarization of multiple documents"""
    
    def __init__(self):
        # self.llm = ChatOpenAI(
        #     model=Config.MODEL_NAME,
        #     temperature=Config.TEMPERATURE,
        #     max_tokens=Config.MAX_TOKENS,
        #     openai_api_key=Config.OPENAI_API_KEY
        # )

        self.llm = init_chat_model(Config.MODEL_NAME)
        
        self.multi_doc_summary_prompt = PromptTemplate(
            input_variables=["documents_info", "combined_text_sample"],
            template="""
            You are analyzing a collection of {num_documents} documents. Provide a comprehensive analysis.
            
            Document Collection Overview:
            {documents_info}
            
            Combined Content Sample (first 4000 characters):
            {combined_text_sample}
            
            Please provide:
            1. **Collection Summary**: Overall summary of all documents and their main themes
            2. **Individual Document Insights**: Brief summary of what each document contributes
            3. **Common Themes**: Topics or themes that appear across multiple documents
            4. **Document Relationships**: How the documents relate to each other (complementary, contrasting, etc.)
            5. **Key Findings**: Most important insights from the entire collection
            
            Format your response clearly with the sections above.
            """
        )
        
        self.multi_doc_questions_prompt = PromptTemplate(
            input_variables=["combined_summary", "documents_info"],
            template="""
            Based on this collection of {num_documents} documents, generate 8 diverse questions that users might ask.
            
            Collection Summary:
            {combined_summary}
            
            Documents in Collection:
            {documents_info}
            
            Generate questions that:
            1. Cover different documents in the collection
            2. Ask about relationships between documents
            3. Explore common themes across documents  
            4. Include both specific and analytical questions
            5. Range from factual to comparative questions
            
            Provide exactly 8 questions in this format:
            1. [Question about specific document]
            2. [Question comparing documents]
            3. [Question about common themes]
            4. [Question about key insights]
            5. [Question about document relationships]
            6. [Question about specific details]
            7. [Question about implications/conclusions]
            8. [Question about document collection as a whole]
            """
        )
    
    def generate_combined_summary(self, state: MultiFileDocumentState) -> str:
        """Generate summary for the entire document collection"""
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
            combined_text_sample = combined_text[:4000] if len(combined_text) > 4000 else combined_text
            
            # Generate summary
            prompt = self.multi_doc_summary_prompt.format(
                num_documents=len(docs_info),
                documents_info=documents_info,
                combined_text_sample=combined_text_sample
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating combined summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def generate_collection_questions(self, combined_summary: str, state: MultiFileDocumentState) -> List[str]:
        """Generate questions for the document collection"""
        try:
            files = state["files"]
            
            # Prepare document information
            docs_info = []
            for file_id in state["file_upload_order"]:
                file_info = files[file_id]
                if file_info.processing_status == ProcessingStatus.OCR_COMPLETE:
                    docs_info.append(f"- {file_info.file_name} ({file_info.file_type.upper()})")
            
            documents_info = "\n".join(docs_info)
            
            # Generate questions
            prompt = self.multi_doc_questions_prompt.format(
                num_documents=len(docs_info),
                combined_summary=combined_summary,
                documents_info=documents_info
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse questions
            questions = self._parse_questions(response.content)
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return [
                "What are the main topics covered across all documents?",
                "How do these documents relate to each other?",
                "What are the key findings from this document collection?",
                "Which document provides the most detailed information on [topic]?",
                "Are there any contradictions between the documents?",
                "What common themes appear in multiple documents?",
                "What unique insights does each document provide?",
                "What conclusions can be drawn from this collection?"
            ]
    
    def _parse_questions(self, response_text: str) -> List[str]:
        """Parse questions from LLM response"""
        questions = []
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering and clean up
                question = line.split('.', 1)[-1].strip() if '.' in line else line.lstrip('- •1234567890').strip()
                if question and ('?' in question or len(question) > 10):
                    # Ensure it ends with question mark
                    if not question.endswith('?'):
                        question += '?'
                    questions.append(question)
        
        return questions[:8]  # Return max 8 questions

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
        logger.info("Generating combined summary for all documents...")
        combined_summary = summarizer.generate_combined_summary(state)
        
        # Generate questions
        logger.info("Generating suggested questions...")
        questions = summarizer.generate_collection_questions(combined_summary, state)
        
        # Analyze document relationships (basic implementation)
        # relationships = analyze_document_relationships(state)
        
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

# def analyze_document_relationships(state: MultiFileDocumentState) -> Dict[str, Any]:
#     """Analyze relationships between documents (basic implementation)"""
#     try:
#         files = state["files"]
#         relationships = {
#             "file_types": {},
#             "size_distribution": {},
#             "processing_status": {}
#         }
        
#         # Analyze file types
#         for file_info in files.values():
#             file_type = file_info.file_type
#             relationships["file_types"][file_type] = relationships["file_types"].get(file_type, 0) + 1
        
#         # Analyze size distribution
#         sizes = [f.file_size for f in files.values()]
#         if sizes:
#             relationships["size_distribution"] = {
#                 "total_size": sum(sizes),
#                 "average_size": sum(sizes) / len(sizes),
#                 "largest_file": max(sizes),
#                 "smallest_file": min(sizes)
#             }
        
#         # Processing status
#         for status in ProcessingStatus:
#             count = sum(1 for f in files.values() if f.processing_status == status)
#             if count > 0:
#                 relationships["processing_status"][status.value] = count
        
#         return relationships
        
#     except Exception as e:
#         logger.error(f"Error analyzing relationships: {str(e)}")
#         return {}
