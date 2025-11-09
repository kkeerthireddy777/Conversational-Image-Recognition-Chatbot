"""
Run all tests in sequence to identify issues
"""
import sys
sys.path.append('..')

from test_yolo import test_yolo
from test_blip import test_blip
from test_llm import test_llm
from test_integration import test_integration

def run_all_tests():
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60 + "\n")
    
    results = {}
    
    # Test 1: YOLO
    print("\n[1/4] Testing YOLO...")
    results['YOLO'] = test_yolo()
    input("\nPress Enter to continue to BLIP test...")
    
    # Test 2: BLIP
    print("\n[2/4] Testing BLIP-2...")
    results['BLIP'] = test_blip()
    input("\nPress Enter to continue to LLM test...")
    
    # Test 3: LLM
    print("\n[3/4] Testing LLM...")
    results['LLM'] = test_llm()
    input("\nPress Enter to continue to Integration test...")
    
    # Test 4: Integration
    print("\n[4/4] Testing Integration...")
    results['Integration'] = test_integration()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
    
    print("="*60)
    
    all_passed = all(results.values())
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED - Check output above")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()