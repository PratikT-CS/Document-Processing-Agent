import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from .graph_state import DocumentProcessingState
from .document_processor import process_uploaded_file
from .text_extractor import extract_text_from_document
from .summarizer import generate_document_summary_and_questions
from .qa_agent import process_user_question

logger = logging.getLogger(__name__)

class DocumentProcessingWorkflow:
    """Main LangGraph workflow for document processing and QA"""
    
    def __init__(self):
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Create workflow graph
        workflow = StateGraph(DocumentProcessingState)
        
        # Add nodes
        workflow.add_node("process_file", process_uploaded_file)
        workflow.add_node("extract_text", extract_text_from_document)
        workflow.add_node("generate_summary", generate_document_summary_and_questions)
        workflow.add_node("answer_question", process_user_question)
        
        # Define entry point
        workflow.set_entry_point("process_file")
        
        # Add conditional edges based on processing status
        workflow.add_conditional_edges(
            "process_file",
            self._decide_after_file_processing,
            {
                "continue": "extract_text",
                "answer_question": "answer_question", 
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "extract_text", 
            self._decide_after_text_extraction,
            {
                "continue": "generate_summary",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "generate_summary",
            self._decide_after_summarization,
            {
                "ready": END,
                "error": END
            }
        )
        
        # QA node can be called separately
        workflow.add_edge("answer_question", END)

        import webbrowser
        compiled_workflow = workflow.compile()
        with open("workflow.png", "wb") as f:
            f.write(compiled_workflow.get_graph().draw_mermaid_png())
        
        webbrowser.open("workflow.png")  # Will open in default image viewer
        
        return workflow
    
    def _decide_after_file_processing(self, state: DocumentProcessingState) -> str:
        """Decide next step after file processing"""
        if state.get("processing_status") == "error":
            return "error"
        if state.get("processing_progress") == 100:
            return "answer_question"
        return "continue"
    
    def _decide_after_text_extraction(self, state: DocumentProcessingState) -> str:
        """Decide next step after text extraction"""
        if state.get("processing_status") == "error":
            return "error"
        return "continue"
    
    def _decide_after_summarization(self, state: DocumentProcessingState) -> str:
        """Decide next step after summarization"""
        if state.get("processing_status") == "error":
            return "error"
        return "ready"
    
    def process_document(self, file_path: str, file_name: str) -> Dict[str, Any]:
        """
        Process a document through the complete workflow
        """
        try:
            # Initialize state
            initial_state = DocumentProcessingState()
            initial_state["uploaded_file_path"] = file_path
            initial_state["file_name"] = file_name
            initial_state["processing_status"] = "idle"
            
            logger.info(f"Starting document processing workflow for: {file_name}")
            
            # Run the workflow
            result = self.app.invoke(initial_state)
            
            logger.info(f"Workflow completed with status: {result.get('processing_status')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in document processing workflow: {str(e)}")
            return {
                "processing_status": "error",
                "error_message": f"Workflow error: {str(e)}"
            }
    
    def ask_question(self, state: DocumentProcessingState, question: str) -> Dict[str, Any]:
        """
        Ask a question about the processed document
        """
        try:
            # Update state with question
            state["current_query"] = question
            
            logger.info(f"Processing question: {question[:100]}...")
            
            # Run QA node directly
            result = self.app.invoke(state, config={"configurable": {"start": "answer_question"}})
            
            logger.info("Question answered successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in QA workflow: {str(e)}")
            state["response"] = f"Error processing question: {str(e)}"
            return state
    
    def get_processing_status(self, state: DocumentProcessingState) -> Dict[str, Any]:
        """Get current processing status"""
        return {
            "status": state.get("processing_status", "idle"),
            "step": state.get("current_step", "upload"),
            "progress": state.get("processing_progress", 0),
            "error": state.get("error_message"),
            "ready_for_qa": state.get("processing_status") == "summarized"
        }

# Singleton instance
document_workflow = DocumentProcessingWorkflow()

def get_workflow() -> DocumentProcessingWorkflow:
    """Get the document processing workflow instance"""
    return document_workflow