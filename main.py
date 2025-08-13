"""
Document Processing Chatbot - Main Entry Point
"""
import sys
import os
import logging
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.ui.gradio_interface import launch_app
from src.config.settings import Config

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('document_chatbot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main application entry point"""
    print("🚀 Starting Document Processing Chatbot...")
    
    # Setup
    setup_logging()
    Config.ensure_directories()
    
    # Check required environment variables
    if not Config.GEMINI_API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable is not set!")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    try:
        # Launch Gradio app
        print("🌐 Launching Gradio interface...")
        launch_app()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()