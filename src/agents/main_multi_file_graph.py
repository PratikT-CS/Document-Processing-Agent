import logging
from typing import Dict, List, Tuple, Any
from langgraph.graph import StateGraph, END
from .multi_file_state import MultiFileDocumentState, ProcessingStatus
from .multi_file_processor import upload_multiple_files, process_all_files_ocr
from .multi_file_summarizer import generate_multi_document_summary
from .multi_file_qa import process_multi_document_question

logger = logging.getLogger(__name__)

class MultiFileDocumentWorkflow:
    """Main LangGraph workflow for multi-file document processing and QA"""
    
    def __init__(self):
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Create workflow graph
        workflow = StateGraph(MultiFileDocumentState)
        
        # Add nodes
        workflow.add_node("upload_files", upload_multiple_files)
        workflow.add_node("process_ocr", process_all_files_ocr)
        workflow.add_node("generate_summary", generate_multi_document_summary)
        workflow.add_node("answer_question", process_multi_document_question)
        
        # Define entry point
        workflow.set_entry_point("upload_files")
        
        # Add conditional edges based on processing status

        workflow.add_conditional_edges(
            "upload_files",
            self._decide_after_upload,
            {
                "continue": "process_ocr",
                "answer_question": "answer_question",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "process_ocr", 
            self._decide_after_ocr,
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
        with open("workflow_multi_file.png", "wb") as f:
            f.write(compiled_workflow.get_graph().draw_mermaid_png())
        
        webbrowser.open("workflow_multi_file.png")  # Will open in default image viewer
        
        return workflow
    
    def _decide_after_upload(self, state: MultiFileDocumentState) -> str:
        """Decide next step after file upload"""
        if state.get("overall_status") == ProcessingStatus.ERROR:
            print(f"Status: {state.get('overall_status')}")
            return "error"
        if state.get("overall_status") == ProcessingStatus.SUMMARIZED:
            print(f"Status: {state.get('overall_status')}")
            return "answer_question"
        print(f"Status: {state.get('overall_status')}")
        return "continue"
    
    def _decide_after_ocr(self, state: MultiFileDocumentState) -> str:
        """Decide next step after OCR processing"""
        if state.get("overall_status") == "error":
            return "error"
        return "continue"
    
    def _decide_after_summarization(self, state: MultiFileDocumentState) -> str:
        """Decide next step after summarization"""
        if state.get("overall_status") == "error":
            return "error"
        return "ready"
    
    def process_documents(self, uploaded_files: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Process multiple documents through the complete workflow
        """
        try:
            # Initialize state
            initial_state = MultiFileDocumentState()
            initial_state["uploaded_file_paths"] = uploaded_files
            initial_state["overall_status"] = ProcessingStatus.IDLE
            
            logger.info(f"Starting multi-file document processing workflow for {len(uploaded_files)} files")
            
            # Run the workflow
            result = self.app.invoke(initial_state)
            
            logger.info(f"Workflow completed with status: {result.get('overall_status')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-file document processing workflow: {str(e)}")
            return {
                "overall_status": "error",
                "error_message": f"Workflow error: {str(e)}"
            }
    
    def ask_question(self, state: MultiFileDocumentState, question: str) -> Dict[str, Any]:
        """
        Ask a question about the processed documents
        """
        try:
            # Update state with question
            state["current_query"] = question
            
            logger.info(f"Processing question: {question}...")
            
            # Run QA node directly
            result = self.app.invoke(state, config={"configurable": {"start": "answer_question"}})
            
            logger.info("Question answered successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in QA workflow: {str(e)}")
            state["response"] = f"Error processing question: {str(e)}"
            return state
    
    def get_processing_status(self, state: MultiFileDocumentState) -> Dict[str, Any]:
        """Get current processing status"""
        return {
            "status": state.get("overall_status", "idle"),
            "step": state.get("current_step", "upload"),
            "progress": state.get("processing_progress", {}),
            "error": state.get("error_message"),
            "ready_for_qa": state.get("overall_status") == "summarized"
        }

# Singleton instance
multi_file_workflow = MultiFileDocumentWorkflow()

def get_workflow() -> MultiFileDocumentWorkflow:
    """Get the multi-file document processing workflow instance"""
    return multi_file_workflow