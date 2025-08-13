import gradio as gr

def chat_with_files(message, history, files):
    # Process files
    file_info = ""
    if files:
        file_info = f"\n[{len(files)} file(s) uploaded]"
    
    # Add to history
    history.append([message + file_info, f"I received: {message}"])
    
    # Process files here (read content, analyze, etc.)
    if files:
        for file in files:
            # Example: read text file
            if file.name.endswith('.txt'):
                with open(file.name, 'r') as f:
                    content = f.read()[:200]  # First 200 chars
                    history.append([None, f"File content preview: {content}..."])
    
    return history, ""

with gr.Blocks() as demo:
    gr.Markdown("# Chat with File Upload")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Message")
    files = gr.File(label="Upload Files", file_count="multiple")
    
    def respond(message, chat_history, uploaded_files):
        return chat_with_files(message, chat_history, uploaded_files)
    
    msg.submit(respond, [msg, chatbot, files], [chatbot, msg])

demo.launch()