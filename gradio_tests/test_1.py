import gradio as gr

def respond_with_file(message, history, file):
    if file is not None:
        print("File Present")
        # Process the uploaded file
        file_info = f"File uploaded: {file.name if hasattr(file, 'name') else 'Unknown'}"
        response = f"I received your message: '{message}' and your file: {file_info}"
    else:
        print("File NOT Present")
        response = f"I received your message: '{message}' (no file uploaded)"
    
    return response

# Create the chat interface with file upload
demo = gr.ChatInterface(
    respond_with_file,
    additional_inputs=[
        gr.File(label="Upload a file", file_count="multiple")
    ],
    title="Chat with File Upload",
    multimodal=True,
    # examples=["Hey, There!", "I want you to assist me on my tasks today."]
)

demo.launch()