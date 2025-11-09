import gradio as gr
from main import ConversationalImageChatbot
import cv2
import numpy as np

# Initialize chatbot
chatbot = ConversationalImageChatbot()

def process_image(image):
    """Handle new image upload"""
    if image is None:
        return None, "Please upload an image first.", [], ""
    
    try:
        # Save temporary image
        temp_path = "temp_image.jpg"
        
        # Handle different image input types
        if isinstance(image, np.ndarray):
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            image.save(temp_path)
        
        print("Processing image...")
        
        # Process image
        response = chatbot.process_new_image(temp_path)
        
        print("Getting detection visualization...")
        # Get annotated image
        annotated = chatbot.get_detection_visualization()
        
        # Get initial history
        history_text = show_conversation_history()
        
        print("Image processing complete!")
        
        # Return with empty chat history for new image
        return annotated, response, [], history_text
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        return None, error_msg, [], ""

def chat_with_image(message, history):
    """Handle chat messages"""
    if not message or message.strip() == "":
        return history, "", show_conversation_history()
    
    try:
        response = chatbot.chat(message)
        history.append((message, response))
        
        # Update history display
        history_text = show_conversation_history()
        
        return history, "", history_text
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append((message, error_msg))
        return history, "", show_conversation_history()

def show_conversation_history():
    """Display conversation history"""
    try:
        history = chatbot.llm.get_conversation_history()
        if not history:
            return "No conversation history yet. Start chatting about the image!"
        
        formatted = []
        for i, msg in enumerate(history, 1):
            role = "👤 You" if msg["role"] == "user" else "🤖 Bot"
            formatted.append(f"**Message {i}** - {role}:\n{msg['content']}\n")
        
        return "\n---\n\n".join(formatted)
    except Exception as e:
        return f"Error loading history: {str(e)}"

def clear_chat():
    """Clear the chat interface"""
    return [], "", "Chat cleared. Conversation history reset."

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Image Recognition Chatbot") as demo:
    gr.Markdown("""
    # 🤖 Conversational Image Recognition Chatbot
    ### Upload an image and have a natural conversation about it!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="numpy")
            upload_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            annotated_output = gr.Image(label="Detected Objects with Bounding Boxes")
    
    initial_response = gr.Textbox(
        label="🎯 Initial Analysis", 
        lines=4,
        placeholder="Initial analysis will appear here after image upload..."
    )
    
    gr.Markdown("---")
    gr.Markdown("## 💬 Chat About The Image")
    
    chatbot_interface = gr.Chatbot(
        label="Conversation", 
        height=400,
        placeholder="Upload an image and start asking questions..."
    )
    
    with gr.Row():
        msg_input = gr.Textbox(
            label="Your message",
            placeholder="Ask anything about the image...",
            scale=4,
            lines=2
        )
        with gr.Column(scale=1):
            send_btn = gr.Button("📤 Send", variant="primary")
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
    
    gr.Markdown("---")
    
    with gr.Accordion("📜 View Conversation History", open=False):
        history_output = gr.Textbox(
            label="Complete Chat History", 
            lines=12,
            placeholder="Conversation history will appear here...",
            show_copy_button=True
        )
        refresh_history_btn = gr.Button("🔄 Refresh History", variant="secondary")
    
    with gr.Accordion("💡 Example Questions", open=True):
        gr.Markdown("""
        **Spatial Questions:**
        - "Where is the [object]?"
        - "How many [objects] are there?"
        - "What's in the top right corner?"
        
        **Visual Details:**
        - "What color is the [object]?"
        - "What is the person wearing?"
        - "What is happening in the image?"
        
        **General Questions:**
        - "Describe the scene"
        - "What's the mood of this image?"
        """)
    
    # Event handlers - FIXED to prevent infinite loops
    upload_btn.click(
        fn=process_image,
        inputs=[image_input],
        outputs=[annotated_output, initial_response, chatbot_interface, history_output],
        show_progress=True
    )
    
    send_btn.click(
        fn=chat_with_image,
        inputs=[msg_input, chatbot_interface],
        outputs=[chatbot_interface, msg_input, history_output],
        show_progress=True
    )
    
    msg_input.submit(
        fn=chat_with_image,
        inputs=[msg_input, chatbot_interface],
        outputs=[chatbot_interface, msg_input, history_output],
        show_progress=True
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot_interface, msg_input, history_output]
    )
    
    refresh_history_btn.click(
        fn=show_conversation_history,
        outputs=[history_output]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        debug=True
    )