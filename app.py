import gradio as gr
from main import ConversationalImageChatbot
import cv2

# Initialize chatbot
chatbot = ConversationalImageChatbot()

def process_image(image):
    """Handle new image upload"""
    if image is None:
        return None, "Please upload an image first.", []
    
    # Save temporary image
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Process image
    response = chatbot.process_new_image(temp_path)
    
    # Get annotated image
    annotated = chatbot.get_detection_visualization()
    
    # Return with empty chat history for new image
    return annotated, response, []

def chat_with_image(message, history):
    """Handle chat messages"""
    if message.strip() == "":
        return history, ""
    
    response = chatbot.chat(message)
    history.append((message, response))
    return history, ""

def show_conversation_history():
    """Display conversation history"""
    history = chatbot.llm.get_conversation_history()
    if not history:
        return "No conversation history yet. Start chatting about the image!"
    
    formatted = []
    for i, msg in enumerate(history, 1):
        role = "👤 You" if msg["role"] == "user" else "🤖 Bot"
        formatted.append(f"**Message {i}** - {role}:\n{msg['content']}\n")
    
    return "\n---\n\n".join(formatted)

def clear_chat():
    """Clear the chat interface"""
    return [], ""

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Image Recognition Chatbot") as demo:
    gr.Markdown("""
    # 🤖 Conversational Image Recognition Chatbot
    ### Upload an image and have a natural conversation about it!
    This chatbot uses YOLOv8 for object detection, BLIP-2 for image understanding, and LLaMA 3 for natural conversations.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="numpy")
            upload_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
            gr.Markdown("""
            **Tip:** Upload any image to start. The system will:
            - Detect objects with bounding boxes
            - Generate a scene description
            - Enable natural conversation about the image
            """)
            
        with gr.Column(scale=1):
            annotated_output = gr.Image(label="Detected Objects with Bounding Boxes")
            gr.Markdown("""
            **Detection Info:** Bounding boxes show detected objects with labels and confidence scores.
            """)
    
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
            placeholder="Ask anything about the image... (e.g., 'Where is the cat?', 'How many people?')",
            scale=4,
            lines=2
        )
        with gr.Column(scale=1):
            send_btn = gr.Button("📤 Send", variant="primary")
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
    
    gr.Markdown("---")
    
    with gr.Accordion("📜 View Conversation History", open=False):
        gr.Markdown("""
        **See the complete conversation thread** including all questions and responses.
        This helps track the context and flow of your discussion about the image.
        """)
        history_output = gr.Textbox(
            label="Complete Chat History", 
            lines=12,
            placeholder="Conversation history will appear here...",
            show_copy_button=True
        )
        refresh_history_btn = gr.Button("🔄 Refresh History", variant="secondary")
    
    gr.Markdown("---")
    
    with gr.Accordion("💡 Example Questions You Can Ask", open=True):
        gr.Markdown("""
        ### Spatial Questions (Uses YOLO Detection):
        - "Where is the [object] in the image?"
        - "How many [objects] are there?"
        - "What's in the top right corner?"
        - "What objects are on the left side?"
        - "Is there a [object] in the center?"
        
        ### Visual Details (Uses BLIP-2 VQA):
        - "What color is the [object]?"
        - "What is the person wearing?"
        - "What is the person doing?"
        - "What's the expression on their face?"
        - "Describe the background"
        
        ### General Questions (Uses LLM):
        - "Tell me more about this scene"
        - "What's the mood of this image?"
        - "What time of day does this look like?"
        - "What can you tell me about the setting?"
        - "Is this indoors or outdoors?"
        """)
    
    with gr.Accordion("ℹ️ How It Works", open=False):
        gr.Markdown("""
        ### Technology Stack:
        
        1. **YOLOv8** - Object Detection
           - Identifies objects in the image
           - Provides precise locations (bounding boxes)
           - Determines spatial positions (top-left, center, etc.)
        
        2. **BLIP-2** - Image Understanding
           - Generates scene descriptions
           - Answers visual questions about colors, actions, details
           - Provides deep visual comprehension
        
        3. **LLaMA 3 (70B)** - Conversational AI
           - Maintains conversation context
           - Generates natural responses
           - Combines detection + vision insights
        
        4. **LangGraph** - Memory Management
           - Maintains conversation history
           - Tracks context across messages
           - Enables multi-turn dialogues
        """)
    
    # Event handlers
    upload_btn.click(
        fn=process_image,
        inputs=[image_input],
        outputs=[annotated_output, initial_response, chatbot_interface]
    ).then(
        fn=show_conversation_history,
        outputs=[history_output]
    )
    
    send_btn.click(
        fn=chat_with_image,
        inputs=[msg_input, chatbot_interface],
        outputs=[chatbot_interface, msg_input]
    ).then(
        fn=show_conversation_history,
        outputs=[history_output]
    )
    
    msg_input.submit(
        fn=chat_with_image,
        inputs=[msg_input, chatbot_interface],
        outputs=[chatbot_interface, msg_input]
    ).then(
        fn=show_conversation_history,
        outputs=[history_output]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot_interface, msg_input]
    )
    
    refresh_history_btn.click(
        fn=show_conversation_history,
        outputs=[history_output]
    )
    
    gr.Markdown("""
    ---
    ### 📝 Tips for Best Results:
    - Upload clear, well-lit images for better detection
    - Ask specific questions for more precise answers
    - Use "where" questions to leverage spatial detection
    - The chatbot remembers context within each image session
    - Upload a new image to start a fresh conversation
    
    **Made with ❤️ using YOLOv8, BLIP-2, LLaMA 3, and LangGraph**
    """)

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )