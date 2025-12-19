import cv2
import os
import sys

print("1. Testing imports...")
try:
    from hairline_detector import HairlineDetector
    print("✅ HairlineDetector imported")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

print("2. Testing data manager...")
try:
    from data.data_manager import DataManager
    dm = DataManager()
    print("✅ DataManager working")
except Exception as e:
    print(f"❌ DataManager error: {e}")
    sys.exit(1)

print("3. Testing with real face image...")
try:
    detector = HairlineDetector()
    print("✅ HairlineDetector initialized")
    
    # Test on real face images
    test_images = [
        "data/input/raw_images/kamal.jpg",
        "data/input/raw_images/kam.jpg", 
        "data/input/raw_images/download.jpg"
    ]
    
    for test_image in test_images:
        if os.path.exists(test_image):
            print(f"🔍 Testing: {test_image}")
            image = cv2.imread(test_image)
            if image is not None:
                print(f"✅ Image loaded: {image.shape}")
                result = detector.analyze_hairline(image)
                if result:
                    print("✅ Analysis successful!")
                    print(f"   Hairline type: {result['hairline_type']}")
                    print(f"   Height: {result['hairline_height']:.3f}")
                    
                    # Show visualization
                    vis_image = detector.visualize_analysis(image, result)
                    cv2.imshow('Hairline Analysis', vis_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    break
                else:
                    print("❌ Analysis failed - no face detected")
            else:
                print("❌ Could not load image")
        else:
            print(f"❌ Test image not found: {test_image}")
        
except Exception as e:
    print(f"❌ Hairline detector error: {e}")

print("🎯 DEBUG COMPLETED")