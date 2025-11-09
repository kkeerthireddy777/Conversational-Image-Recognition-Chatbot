"""
Test LLM conversational capabilities independently
"""
import sys
sys.path.append('..')

from models.llm_conversational import ConversationalLLM
import time

def test_llm():
    print("="*60)
    print("TESTING LLM CONVERSATION")
    print("="*60)
    
    try:
        # Initialize LLM
        print("\n1. Initializing LLM...")
        start = time.time()
        llm = ConversationalLLM()
        print(f"   ✓ LLM loaded in {time.time()-start:.2f}s")
        
        # Mock image context
        mock_context = """
Overall Scene Description: A cat sitting on a couch in a living room

Detected: 1 cat at center of the image; 1 couch at bottom center.

Detailed Object Positions:
1. cat - located at center of the image (confidence: 0.89)
2. couch - located at bottom center (confidence: 0.76)
        """
        
        # Test conversation flow
        print("\n2. Testing Conversation Flow...")
        
        test_queries = [
            "What do you see in this image?",
            "Where is the cat?",
            "How many objects are there?",
            "What is the cat doing?",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Turn {i}:")
            print(f"   User: {query}")
            start = time.time()
            response = llm.generate_response(query, mock_context)
            print(f"   Bot: {response}")
            print(f"   (Response time: {time.time()-start:.2f}s)")
        
        # Test conversation history
        print("\n3. Testing Conversation History...")
        history = llm.get_conversation_history()
        print(f"   Total messages in history: {len(history)}")
        for i, msg in enumerate(history, 1):
            role = msg['role']
            content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
            print(f"   {i}. [{role}] {content}")
        
        # Test memory reset
        print("\n4. Testing Memory Reset...")
        llm.reset_memory()
        history_after_reset = llm.get_conversation_history()
        print(f"   History after reset: {len(history_after_reset)} messages")
        
        print("\n" + "="*60)
        print("✅ LLM TEST PASSED")
        print("="*60)
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ LLM TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_llm()