import cv2
import os
from hairline_detector import HairlineDetector

# Your image path
image_path = r"C:\Users\davii\OneDrive\Desktop\ai proj\data\input\raw_images\kam.jpg"

print(f"🔍 Testing your image: {image_path}")

# Check if file exists
if not os.path.exists(image_path):
    print("❌ Image file not found!")
    print("Please check the path and make sure the file exists")
else:
    # Initialize detector
    detector = HairlineDetector()
    
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Could not load image file")
    else:
        print(f"✅ Image loaded successfully: {image.shape}")
        
        # Analyze hairline
        print("🔄 Analyzing hairline...")
        result = detector.analyze_hairline(image)
        
        if result:
            print("🎉 SUCCESS! Hairline analysis working!")
            print(f"   Hairline Type: {result['hairline_type']}")
            print(f"   Height: {result['hairline_height']:.3f}")
            print(f"   Forehead Ratio: {result['forehead_ratio']:.3f}")
            print(f"   Density Score: {result['density_score']:.3f}")
            print(f"   Symmetry Score: {result['symmetry_score']:.3f}")
            
            # Show visualization
            vis_image = detector.visualize_analysis(image, result)
            cv2.imshow('Hairline Analysis - YOUR IMAGE', vis_image)
            print("📊 Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save results
            from data.data_manager import DataManager
            dm = DataManager()
            dm.save_analysis_result(result, "kamal")
            dm.save_visualization(vis_image, "kamal")
            
            print("💾 Results saved to output folders!")
        else:
            print("❌ No face detected in your image")
            print("This could be because:")
            print("- The face is not clearly visible")
            print("- The image is too dark/bright")
            print("- The face is at an angle")
            print("- Image quality is poor")

print("🎯 Test completed!")