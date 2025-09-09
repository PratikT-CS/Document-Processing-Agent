import gradio as gr

def respond_with_file(message, history, file):
    if file is not None:
        print("File Present")
        # Process the uploaded file
        file_info = f"File uploaded: {file.name if hasattr(file, 'name') else 'Unknown'}"
        response = f"I received your message: '{message}' and your file: {file_info}"
    else:
        print("File NOT Present")
        response = {
                "role": "assistant",
                "content": gr.Image(
                    value=-"https://doc-processing-agent-k.s3.amazonaws.com/cropped_imgs/79b30c9d-e8c4-4c25-8d75-ef9e7c543904.png",
                    label="MV-1.pdf",
                    show_label=True    
                )
            }
        
    
    return response

# Create the chat interface with file upload
with gr.Blocks(title="Document Processing Agent",css=".btn {height : 60px;}") as demo:
    gr.Markdown("# LangGraph Document Agent")
    gr.Markdown('''
        <p>Label</p>
        <img src="https://doc-processing-agent-k.s3.us-east-1.amazonaws.com/cropped_imgs/027026a1-1582-4b1f-b858-995663f7248e.png" style="height: 1rem; width: 1rem;">
        
        **Text**
        Label
        ![Extracted Img](https://doc-processing-agent-k.s3.us-east-1.amazonaws.com/cropped_imgs/027026a1-1582-4b1f-b858-995663f7248e.png "From MV-1.pdf")
        
        Label
        ![Extracted Img](https://doc-processing-agent-k.s3.amazonaws.com/cropped_imgs/79b30c9d-e8c4-4c25-8d75-ef9e7c543904.png "From MV-1.pdf")
    ''')

demo.launch()