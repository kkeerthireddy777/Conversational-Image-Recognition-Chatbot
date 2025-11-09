from models.yolo_detector import YOLODetector
from models.blip_captioner import BLIPCaptioner
from models.llm_conversational import ConversationalLLM
from utils.image_processor import ImageProcessor
from utils.prompt_builder import PromptBuilder
import os

class ConversationalImageChatbot:
    def __init__(self):
        """Initialize all components"""
        print("Initializing Conversational Image Chatbot...")
        
        self.yolo = YOLODetector()
        print("✓ YOLO detector loaded")
        
        self.blip = BLIPCaptioner()
        print("✓ BLIP-2 captioner loaded")
        
        self.llm = ConversationalLLM()
        print("✓ LLM conversation model loaded")
        
        self.image_processor = ImageProcessor()
        self.prompt_builder = PromptBuilder()
        
        self.current_image_context = None
        self.current_image_path = None
        
        print("\n🎉 Chatbot ready!\n")
    
    def process_new_image(self, image_path: str) -> str:
        """
        Process a new image and generate initial analysis
        """
        # Validate image
        if not self.image_processor.validate_image(image_path):
            return "Error: Invalid image file."
        
        # Preprocess image
        processed_path = self.image_processor.preprocess_image(image_path)
        self.current_image_path = processed_path
        
        # Reset conversation for new image
        self.llm.reset_memory()
        
        print("Analyzing image...")
        
        # Run YOLO detection
        print("- Running object detection...")
        yolo_results = self.yolo.detect_objects(processed_path)
        
        # Generate BLIP caption
        print("- Generating image caption...")
        blip_caption = self.blip.generate_caption(processed_path)
        
        # Build context
        self.current_image_context = self.prompt_builder.build_image_context(
            yolo_results, blip_caption
        )
        
        # Generate initial response
        initial_prompt = "Provide a brief, natural description of what you see in this image."
        response = self.llm.generate_response(initial_prompt, self.current_image_context)
        
        return response
    
    def chat(self, user_message: str) -> str:
        """
        Continue conversation about the current image
        """
        if self.current_image_context is None:
            return "Please upload an image first."
        
        # Check if question is very specific (might need BLIP VQA)
        if any(word in user_message.lower() for word in ['color', 'wearing', 'doing', 'expression']):
            # Use BLIP VQA for specific visual questions
            blip_answer = self.blip.answer_question(self.current_image_path, user_message)
            # Enhance with LLM
            enhanced_prompt = f"The visual analysis says: '{blip_answer}'. Provide a natural response to: {user_message}"
            response = self.llm.generate_response(enhanced_prompt, self.current_image_context)
        else:
            # Use LLM with context for general questions
            response = self.llm.generate_response(user_message, self.current_image_context)
        
        return response
    
    def get_detection_visualization(self):
        """Return the annotated image with bounding boxes"""
        if self.current_image_path:
            yolo_results = self.yolo.detect_objects(self.current_image_path)
            return yolo_results['annotated_image']
        return None

# CLI Interface
def main():
    chatbot = ConversationalImageChatbot()
    
    print("=" * 60)
    print("CONVERSATIONAL IMAGE CHATBOT")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Upload new image")
        print("2. Ask question about current image")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                response = chatbot.process_new_image(image_path)
                print(f"\n🤖 Bot: {response}")
            else:
                print("❌ Image file not found!")
        
        elif choice == "2":
            if chatbot.current_image_context is None:
                print("❌ Please upload an image first!")
                continue
            
            user_question = input("Your question: ").strip()
            if user_question:
                response = chatbot.chat(user_question)
                print(f"\n🤖 Bot: {response}")
        
        elif choice == "3":
            print("Goodbye! 👋")
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()