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
        
        # Convert annotated image from BGR to RGB for Gradio display
        if annotated is not None:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Get initial history
        history_text = show_conversation_history()
        
        print("Image processing complete!")
        
        # Return with empty chat history for new image
        return annotated, response, [], history_text
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg, [], ""

def chat_with_image(message, history):
    """Handle chat messages"""
    if not message or message.strip() == "":
        return history, "", show_conversation_history()
    
    try:
        print(f"\nUser: {message}")
        response = chatbot.chat(message)
        print(f"Bot: {response}\n")
        
        # Append to chat history
        history.append([message, response])
        
        # Update history display
        history_text = show_conversation_history()
        
        return history, "", history_text
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        history.append([message, error_msg])
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

def refresh_history():
    """Refresh conversation history"""
    return show_conversation_history()

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Image Recognition Chatbot", css="""
    .gradio-container {font-family: 'Arial', sans-serif;}
    footer {display: none !important;}
""") as demo:
    
    gr.Markdown("""
    # 🤖 Conversational Image Recognition Chatbot
    ### Upload an image and have a natural conversation about it!
    **Powered by YOLOv8 + BLIP-2 + LLaMA 3**
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📤 Upload Image", 
                type="numpy",
                height=300
            )
            upload_btn = gr.Button(
                "🔍 Analyze Image", 
                variant="primary", 
                size="lg"
            )
            gr.Markdown("""
            **Instructions:**
            1. Upload any image (JPG, PNG)
            2. Click "Analyze Image"
            3. Start asking questions!
            """)
            
        with gr.Column(scale=1):
            annotated_output = gr.Image(
                label="🎯 Detected Objects",
                height=300
            )
            gr.Markdown("""
            **Detection Info:**
            Bounding boxes show objects with labels and confidence scores.
            """)
    
    initial_response = gr.Textbox(
        label="📋 Initial Analysis", 
        lines=4,
        placeholder="Analysis will appear here...",
        interactive=False
    )
    
    gr.Markdown("---")
    gr.Markdown("## 💬 Chat About The Image")
    
    chatbot_interface = gr.Chatbot(
        label="Conversation", 
        height=400,
        bubble_full_width=False
    )
    
    with gr.Row():
        msg_input = gr.Textbox(
            label="",
            placeholder="Type your question here... (e.g., 'Where is the cat?', 'How many people?')",
            scale=5,
            lines=1,
            max_lines=3
        )
        with gr.Column(scale=1, min_width=100):
            send_btn = gr.Button("📤 Send", variant="primary", size="sm")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
    
    gr.Markdown("---")
    
    with gr.Accordion("📜 View Full Conversation History", open=False):
        gr.Markdown("See the complete conversation thread with all context.")
        history_output = gr.Textbox(
            label="Complete Chat History", 
            lines=10,
            placeholder="History will appear here...",
            show_copy_button=True,
            interactive=False
        )
        refresh_history_btn = gr.Button("🔄 Refresh History", size="sm")
    
    with gr.Accordion("💡 Example Questions You Can Ask", open=True):
        gr.Markdown("""
        | Category | Example Questions |
        |----------|------------------|
        | **Spatial/Location** | • Where is the [object]?<br>• How many [objects] are there?<br>• What's in the top right corner?<br>• Is there anything on the left side? |
        | **Visual Details** | • What color is the [object]?<br>• What is the person wearing?<br>• What expression does the person have?<br>• Describe the background |
        | **Scene Understanding** | • What's happening in this image?<br>• What time of day is it?<br>• Is this indoors or outdoors?<br>• What's the mood of the scene? |
        | **Specific Objects** | • Tell me about the [object]<br>• What can you see near the [object]?<br>• Are there any animals in the image? |
        """)
    
    with gr.Accordion("ℹ️ Technology Stack", open=False):
        gr.Markdown("""
        ### How It Works:
        
        1. **YOLOv8** - Object Detection
           - Identifies and localizes objects
           - Provides bounding boxes and confidence scores
           - Determines spatial positions
        
        2. **BLIP-2** - Image Understanding  
           - Generates natural language descriptions
           - Answers visual questions (colors, actions, details)
           - Deep scene comprehension
        
        3. **LLaMA 3 (70B)** - Conversational AI
           - Generates natural responses
           - Maintains conversation context
           - Combines all insights intelligently
        
        4. **LangGraph** - Memory Management
           - Tracks conversation history
           - Maintains context across turns
           - Enables coherent multi-turn dialogues
        """)
    
    gr.Markdown("---")
    gr.Markdown("""
    <div style='text-align: center; color: #666;'>
    <small>💡 Tip: Upload clear, well-lit images for best results • The bot remembers context within each image session</small>
    </div>
    """)
    
    # Event handlers
    upload_btn.click(
        fn=process_image,
        inputs=[image_input],
        outputs=[annotated_output, initial_response, chatbot_interface, history_output]
    )
    
    send_btn.click(
        fn=chat_with_image,
        inputs=[msg_input, chatbot_interface],
        outputs=[chatbot_interface, msg_input, history_output]
    )
    
    msg_input.submit(
        fn=chat_with_image,
        inputs=[msg_input, chatbot_interface],
        outputs=[chatbot_interface, msg_input, history_output]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot_interface, msg_input, history_output]
    )
    
    refresh_history_btn.click(
        fn=refresh_history,
        outputs=[history_output]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Gradio Interface...")
    print("="*60 + "\n")
    
    demo.queue()  # Enable queue for better handling
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True  # Auto-open browser
    )