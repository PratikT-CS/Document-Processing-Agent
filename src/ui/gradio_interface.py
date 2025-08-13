import gradio as gr
import logging
from typing import List, Tuple, Optional, Dict, Any
import time
from ..agents.main_graph import get_workflow
from ..agents.graph_state import DocumentProcessingState
from ..config.settings import Config
# from src.agents.main_graph import get_workflow
# from src.agents.graph_state import DocumentProcessingState
# from src.config.settings import Config

logger = logging.getLogger(__name__)

class DocumentChatInterface:
    """Gradio interface for document processing and chat"""
    
    def __init__(self):
        self.workflow = get_workflow()
        self.current_state: Optional[DocumentProcessingState] = None
        self.processing = False
    
    def upload_and_process_file(self, file) -> Tuple[str, str, str, List[str], bool]:
        """
        Handle file upload and processing
        Returns: (status_message, summary, progress, questions, chat_visible)
        """
        try:
            if file is None:
                return "Please select a file to upload.", "", "No file selected", [], False
            
            if self.processing:
                return "Currently processing another file. Please wait.", "", "Processing in progress", [], False
            
            self.processing = True
            
            # Start processing
            logger.info(f"Processing uploaded file: {file.name}")
            
            # Process document through workflow
            result = self.workflow.process_document(file.name, file.name)
            
            # Store state
            self.current_state = result
            
            if result.get("processing_status") == "error":
                error_msg = result.get("error_message", "Unknown error occurred")
                self.processing = False
                return f"Error: {error_msg}", "", "Error occurred", [], False
            
            elif result.get("processing_status") == "summarized":
                summary = result.get("document_summary", "No summary available")
                questions = result.get("suggested_questions", [])
                
                # Format summary for display
                formatted_summary = self._format_summary(summary)
                
                self.processing = False
                return (
                    "✅ Document processed successfully! You can now ask questions.",
                    formatted_summary,
                    "Processing complete",
                    questions,
                    True  # Make chat visible
                )
            
            else:
                self.processing = False
                return "Processing completed but status unclear.", "", "Status unclear", [], False
                
        except Exception as e:
            logger.error(f"Error in upload_and_process_file: {str(e)}")
            self.processing = False
            return f"Error processing file: {str(e)}", "", "Error", [], False
    
    def answer_question(self, question: str, chat_history: List[List[str]]) -> Tuple[List[List[str]], str]:
        """
        Handle user question and return updated chat history
        Returns: (updated_chat_history, empty_input)
        """
        try:
            if not self.current_state:
                chat_history.append([question, "Please upload and process a document first."])
                return chat_history, ""
            
            if self.current_state.get("processing_status") != "summarized":
                chat_history.append([question, "Document is not ready for questions yet. Please wait for processing to complete."])
                return chat_history, ""
            
            # Process question through workflow
            result = self.workflow.ask_question(self.current_state, question)
            
            # Get response
            response = result.get("response", "I couldn't generate a response.")
            
            # Update current state
            self.current_state = result
            
            # Add to chat history
            chat_history.append([question, response])
            
            return chat_history, ""  # Clear input
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            chat_history.append([question, f"Error processing question: {str(e)}"])
            return chat_history, ""
    
    def use_suggested_question(self, question: str, chat_history: List[List[str]]) -> Tuple[List[List[str]], str]:
        """Handle clicking on suggested question"""
        return self.answer_question(question, chat_history)
    
    def _format_summary(self, summary: str) -> str:
        """Format summary for better display"""
        if not summary:
            return "No summary available."
        
        # Add some basic formatting
        formatted = summary.replace("**", "").replace("Summary:", "").strip()
        
        # Add metadata if available
        if self.current_state:
            metadata = self.current_state.get("document_metadata", {})
            if metadata:
                meta_text = f"""
**Document Information:**
- Word Count: {metadata.get('word_count', 'Unknown')}
- Estimated Reading Time: {metadata.get('estimated_reading_time', 'Unknown')} minutes
- Paragraphs: {metadata.get('paragraph_count', 'Unknown')}

**Summary:**
{formatted}
"""
                return meta_text
        
        return formatted
    
    def create_interface(self) -> gr.Interface:
        """Create the Gradio interface"""
        
        with gr.Blocks(title="Document Chat Assistant", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("# 📄 Document Chat Assistant")
            gr.Markdown("Upload a document (PDF, Image, DOCX) and chat with it!")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # File upload section
                    with gr.Group():
                        gr.Markdown("## 📤 Upload Document")
                        
                        file_input = gr.File(
                            label="Select Document",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".docx"],
                            file_count="single"
                        )
                        
                        process_btn = gr.Button("Process Document", variant="primary", size="lg")
                        
                        status_output = gr.Textbox(
                            label="Status",
                            value="Ready to upload file",
                            interactive=False
                        )
                        
                        progress_output = gr.Textbox(
                            label="Progress", 
                            value="Waiting for file",
                            interactive=False
                        )
                
                with gr.Column(scale=2):
                    # Document summary section
                    with gr.Group():
                        gr.Markdown("## 📋 Document Summary")
                        
                        summary_output = gr.Markdown(
                            value="Summary will appear here after processing...",
                            label="Document Summary"
                        )
            
            # Suggested questions section (initially hidden)
            with gr.Group(visible=False) as questions_group:
                gr.Markdown("## 💡 Suggested Questions")
                with gr.Row():
                    question_buttons = []
                    for i in range(5):
                        btn = gr.Button("", visible=False, size="sm")
                        question_buttons.append(btn)
            
            # Chat interface (initially hidden)  
            with gr.Group(visible=False) as chat_group:
                gr.Markdown("## 💬 Chat with Document")
                
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_label=False
                )
                
                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Ask a question",
                        placeholder="Type your question here...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Event handlers
            def handle_processing(file):
                result = self.upload_and_process_file(file)
                status, summary, progress, questions, chat_visible = result

                chatbot.examples = [q for q in questions]
                
                # Update question buttons
                question_updates = []
                for i, btn in enumerate(question_buttons):
                    if i < len(questions):
                        question_updates.extend([
                            gr.Button(value=questions[i], visible=True),
                        ])
                    else:
                        question_updates.extend([
                            gr.Button(visible=False),
                        ])
                
                return [
                    status,  # status_output
                    progress,  # progress_output  
                    summary,  # summary_output
                    gr.Group(visible=bool(questions)),  # questions_group
                    gr.Group(visible=chat_visible),  # chat_group
                ] + question_updates
            
            # Process button click
            process_btn.click(
                fn=handle_processing,
                inputs=[file_input],
                outputs=[
                    status_output,
                    progress_output, 
                    summary_output,
                    questions_group,
                    chat_group
                ] + question_buttons
            )
            
            # Chat input handlers
            chat_input.submit(
                fn=self.answer_question,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input]
            )
            
            send_btn.click(
                fn=self.answer_question,
                inputs=[chat_input, chatbot],
                outputs=[chatbot, chat_input]
            )
            
            # Suggested question handlers
            for btn in question_buttons:
                btn.click(
                    fn=lambda q, hist: self.use_suggested_question(q, hist),
                    inputs=[btn, chatbot],
                    outputs=[chatbot, chat_input]
                )
        
        return interface

def create_app() -> gr.Interface:
    """Create and return the Gradio app"""
    Config.ensure_directories()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    interface = DocumentChatInterface()
    return interface.create_interface()

def launch_app():
    """Launch the Gradio application"""
    app = create_app()
    app.launch(
        server_port=Config.GRADIO_PORT,
        share=Config.GRADIO_SHARE,
        debug=True
    )

if __name__ == "__main__":
    launch_app()