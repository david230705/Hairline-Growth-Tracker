import cv2
from hairline_detector import HairlineDetector

# Test with the fixed images
detector = HairlineDetector()

# Try each sample image
for i in range(1, 4):
    image_path = f"data/input/raw_images/sample_{i:02d}.jpg"
    print(f"🔍 Testing: {image_path}")
    
    image = cv2.imread(image_path)
    if image is not None:
        print(f"✅ Image loaded: {image.shape}")
        result = detector.analyze_hairline(image)
        if result:
            print(f"✅ Analysis successful! Hairline type: {result['hairline_type']}")
        else:
            print("❌ Analysis failed - no face detected")
    else:
        print("❌ Could not load image")