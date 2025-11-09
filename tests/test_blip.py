"""
Test BLIP-2 image captioning and VQA independently
"""
import sys
sys.path.append('..')

from models.blip_captioner import BLIPCaptioner
import time

def test_blip():
    print("="*60)
    print("TESTING BLIP-2 CAPTIONER")
    print("="*60)
    
    try:
        # Initialize BLIP
        print("\n1. Initializing BLIP-2...")
        start = time.time()
        blip = BLIPCaptioner()
        print(f"   ✓ BLIP-2 loaded in {time.time()-start:.2f}s")
        
        # Test image path
        image_path = "sample_image.jpg"  # Replace with your test image
        print(f"\n2. Testing with image: {image_path}")
        
        # Test Caption Generation
        print("\n3. Testing Caption Generation...")
        start = time.time()
        caption = blip.generate_caption(image_path)
        print(f"   ✓ Caption generated in {time.time()-start:.2f}s")
        print(f"   Caption: {caption}")
        
        # Test VQA (Visual Question Answering)
        print("\n4. Testing Visual Question Answering...")
        test_questions = [
            "What is in the image?",
            "What color is the main object?",
            "Where is the object located?",
        ]
        
        for q in test_questions:
            print(f"\n   Q: {q}")
            start = time.time()
            answer = blip.answer_question(image_path, q)
            print(f"   A: {answer} ({time.time()-start:.2f}s)")
        
        print("\n" + "="*60)
        print("✅ BLIP-2 TEST PASSED")
        print("="*60)
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ BLIP-2 TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_blip()