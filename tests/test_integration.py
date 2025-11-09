"""
Test full integration of all components
"""
import sys
sys.path.append('..')

from main import ConversationalImageChatbot
import time

def test_integration():
    print("="*60)
    print("TESTING FULL INTEGRATION")
    print("="*60)
    
    try:
        # Initialize chatbot
        print("\n1. Initializing Full Chatbot...")
        start = time.time()
        chatbot = ConversationalImageChatbot()
        print(f"   ✓ Chatbot initialized in {time.time()-start:.2f}s")
        
        # Test image path
        image_path = "sample_image.jpg"  # Replace with your test image
        print(f"\n2. Processing image: {image_path}")
        
        # Process image
        print("   Processing...")
        start = time.time()
        initial_response = chatbot.process_new_image(image_path)
        print(f"   ✓ Image processed in {time.time()-start:.2f}s")
        print(f"\n   Initial Response:")
        print(f"   {initial_response}")
        
        # Test conversation
        print("\n3. Testing Conversation...")
        test_questions = [
            "What objects do you see?",
            "Where is the main object?",
            "How many items are there?",
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n   Q{i}: {question}")
            start = time.time()
            response = chatbot.chat(question)
            print(f"   A{i}: {response}")
            print(f"   (Time: {time.time()-start:.2f}s)")
        
        # Test visualization
        print("\n4. Testing Visualization...")
        annotated = chatbot.get_detection_visualization()
        if annotated is not None:
            import cv2
            cv2.imwrite("test_integration_output.jpg", annotated)
            print(f"   ✓ Annotated image saved to: test_integration_output.jpg")
        else:
            print("   ⚠ No visualization generated")
        
        # Check conversation history
        print("\n5. Checking Conversation History...")
        history = chatbot.llm.get_conversation_history()
        print(f"   Total messages: {len(history)}")
        
        print("\n" + "="*60)
        print("✅ INTEGRATION TEST PASSED")
        print("="*60)
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ INTEGRATION TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_integration()