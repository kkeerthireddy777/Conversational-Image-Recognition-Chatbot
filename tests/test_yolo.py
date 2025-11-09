"""
Test YOLO object detection independently
"""
import sys
sys.path.append('..')

from models.yolo_detector import YOLODetector
import cv2
import time

def test_yolo():
    print("="*60)
    print("TESTING YOLO DETECTOR")
    print("="*60)
    
    try:
        # Initialize YOLO
        print("\n1. Initializing YOLO...")
        start = time.time()
        yolo = YOLODetector()
        print(f"   ✓ YOLO loaded in {time.time()-start:.2f}s")
        
        # Test image path
        image_path = "sample_image.jpg"  # Replace with your test image
        print(f"\n2. Testing with image: {image_path}")
        
        # Detect objects
        print("   Running detection...")
        start = time.time()
        results = yolo.detect_objects(image_path)
        print(f"   ✓ Detection completed in {time.time()-start:.2f}s")
        
        # Print results
        print(f"\n3. Detection Results:")
        print(f"   Total objects: {results['total_objects']}")
        print(f"\n   Structured info:")
        print(f"   {results['structured_info']}")
        
        print(f"\n   Detailed detections:")
        for i, det in enumerate(results['detections'], 1):
            print(f"   {i}. {det['class']} - {det['position']} (conf: {det['confidence']:.2f})")
        
        # Save annotated image
        output_path = "test_yolo_output.jpg"
        cv2.imwrite(output_path, results['annotated_image'])
        print(f"\n4. Annotated image saved to: {output_path}")
        
        print("\n" + "="*60)
        print("✅ YOLO TEST PASSED")
        print("="*60)
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ YOLO TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_yolo()