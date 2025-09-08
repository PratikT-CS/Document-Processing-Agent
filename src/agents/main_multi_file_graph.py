import logging
from typing import Dict, List, Tuple, Any
from langgraph.graph import StateGraph, END, START
from .multi_file_state import MultiFileDocumentState, ProcessingStatus
from .multi_file_processor import collect_bda_results, process_single_bda, upload_multiple_files, process_all_files_ocr
from .multi_file_summarizer import generate_multi_document_summary
from .multi_file_qa import process_multi_document_question, format_response_for_gradio
from .store_embeddings import store_embeddings
from langgraph.prebuilt import ToolNode, tools_condition
from .multi_file_qa import tools
from langgraph.types import Send
from langgraph.checkpoint.postgres import PostgresSaver
import os
from .checkpointer_manager import CheckpointerManager

logger = logging.getLogger(__name__)

class MultiFileDocumentWorkflow:
    """Main LangGraph workflow for multi-file document processing and QA"""
    
    def __init__(self):
        self.workflow = self._create_workflow()
        checkpointer = CheckpointerManager.get_checkpointer()
        self.app = self.workflow.compile(checkpointer=checkpointer)
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Create workflow graph
        workflow = StateGraph(MultiFileDocumentState)
        
        # Add nodes
        workflow.add_node("upload_files", upload_multiple_files)
        workflow.add_node("process_ocr", process_all_files_ocr)
        workflow.add_node("generate_summary", generate_multi_document_summary)
        workflow.add_node("store_embeddings", store_embeddings)
        workflow.add_node("answer_question", process_multi_document_question)
        workflow.add_node("tools", ToolNode(tools=tools))
        workflow.add_node("format_response", format_response_for_gradio)
        workflow.add_node("process_single_bda", process_single_bda)
        workflow.add_node("collect_bda_results", collect_bda_results)
        
        # Define entry point
        workflow.add_conditional_edges(
            START,
            self._route_initial_request,
            {
                "process": "upload_files",
                "QnA": "answer_question"
            }
        )
        
        # Add conditional edges based on processing status
        workflow.add_conditional_edges(
            "upload_files",
            self._route_on_status,
            {
                "continue": "process_ocr",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "process_ocr", 
            self._route_bda,
            {
                "continue": "collect_bda_results",
                "error": END
            }
        )
        
        workflow.add_edge("process_single_bda", "collect_bda_results")
        workflow.add_edge("collect_bda_results", "generate_summary")
        
        workflow.add_conditional_edges(
            "generate_summary",
            self._route_on_status,
            {
                "continue": "store_embeddings",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "store_embeddings",
            self._route_on_status,
            {
                "continue": END,
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "answer_question", 
            self._check_tool_calls, 
            {
                "tool_call": "tools", 
                "end": "format_response"
            }
        )
        workflow.add_edge("tools", "answer_question")
        workflow.add_edge("format_response", END)

        compiled_workflow = workflow.compile()
        with open("workflow_multi_file.png", "wb") as f:
            f.write(compiled_workflow.get_graph().draw_mermaid_png())
        
        return workflow
    
    def _route_initial_request(self, state: MultiFileDocumentState) -> str:
        """Decide whether to proceed with processing or go straight to Q&A"""
        if state.get("uploaded_file_paths") == []:
            return "QnA"
        else:
            return "process"
        
    def _route_on_status(self, state: MultiFileDocumentState) -> str:
        """Decide which branch of the workflow to take based on overall status"""
        if state.get("overall_status") == ProcessingStatus.ERROR:
            return "error"
        return "continue"
    
    def _route_bda(self, state:MultiFileDocumentState) -> str:
        """Route after OCR - either to BDA processing or directly to collect results"""
        if state.get("overall_status") == ProcessingStatus.ERROR:
            return "error"
        
        # Check if any files need BDA processing
        files_to_process = ["bill of sale", "compliance pack", "mv-1", "store pack"]
        
        sends = []
        for file_id, file_info in state["files"].items():
            should_process_with_bda = any(name in file_info.file_name.lower() for name in files_to_process)
            if should_process_with_bda:
                sends.append(Send("process_single_bda", {
                    "file_id": file_id,
                    "file_info": file_info
                }))
        
        if sends:
            return sends  # This will trigger parallel BDA processing
        else:
            return "continue"  # Skip BDA processing
    
    def _check_tool_calls(self, state: MultiFileDocumentState) -> str:
        """Decide tools call is present or not"""
        messages = state.get("messages", [])
        last_msg = messages[-1]
        if last_msg.tool_calls:
            return "tool_call"
        else:
            return "end"
    
    def process_documents(self, uploaded_files: List[Tuple[str, str]], user_id: str) -> Dict[str, Any]:
        """
        Process multiple documents through the complete workflow
        """
        try:
            # Initialize state
            initial_state = MultiFileDocumentState()
            initial_state["uploaded_file_paths"] = uploaded_files
            initial_state["overall_status"] = ProcessingStatus.IDLE
            initial_state["user_id"] = user_id
            
            logger.info(f"Starting multi-file document processing workflow for {len(uploaded_files)} files")
            
            # Run the workflow
            config = {"configurable": {"thread_id": "1", "user_id": user_id}}
            result = self.app.invoke(initial_state, config=config)
            
            logger.info(f"Workflow completed with status: {result.get('overall_status')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-file document processing workflow: {str(e)}")
            return {
                "overall_status": "error",
                "error_message": f"Workflow error: {str(e)}"
            }
    
    def ask_question(self, state: MultiFileDocumentState, question: str, user_id: str) -> Dict[str, Any]:
        """
        Ask a question about the processed documents
        """
        try:
            if not question.strip():
                raise ("Question cannot be empty")
            
            if not state.get("overall_status") == ProcessingStatus.VECTORIZED:
                raise ("Documents must be processed before asking questions")
            
            # Update state with question
            state["current_query"] = question
            
            logger.info(f"Processing question: {question}...")
            
            # Run QA node directly
            config = {"configurable": {"thread_id": "1", "user_id": user_id}}
            result = self.app.invoke(state, config=config)
            
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