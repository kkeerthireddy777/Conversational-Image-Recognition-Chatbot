"""
Minimal Gradio test to check UI without full integration
"""
import gradio as gr
import time

def simple_process(image):
    """Simple image processing test"""
    if image is None:
        return None, "No image uploaded"
    
    time.sleep(1)  # Simulate processing
    return image, "Image processed successfully!"

def simple_chat(message, history):
    """Simple chat test"""
    if not message:
        return history, ""
    
    response = f"Echo: {message}"
    history.append([message, response])
    return history, ""

with gr.Blocks() as demo:
    gr.Markdown("# Minimal Gradio Test")
    
    with gr.Row():
        image_input = gr.Image(label="Upload", type="numpy")
        image_output = gr.Image(label="Output")
    
    process_btn = gr.Button("Process")
    status = gr.Textbox(label="Status")
    
    gr.Markdown("## Chat Test")
    chatbot = gr.Chatbot(height=300)
    
    with gr.Row():
        msg = gr.Textbox(label="Message", scale=4)
        send = gr.Button("Send", scale=1)
    
    # Event handlers
    process_btn.click(
        fn=simple_process,
        inputs=[image_input],
        outputs=[image_output, status]
    )
    
    send.click(
        fn=simple_chat,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        fn=simple_chat,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

if __name__ == "__main__":
    print("Testing minimal Gradio interface...")
    demo.launch(server_name="127.0.0.1", server_port=7860)