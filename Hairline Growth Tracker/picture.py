import cv2
import os
from hairline_detector import HairlineDetector
from data.data_manager import DataManager

def take_and_analyze_photo():
    # Initialize components
    detector = HairlineDetector()
    data_manager = DataManager()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return
    
    print("📷 Webcam activated!")
    print("💡 Instructions:")
    print("   - Look straight at the camera")
    print("   - Make sure your face is well-lit")
    print("   - Press SPACEBAR to capture photo")
    print("   - Press ESC to cancel")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image")
            break
        
        # Display the live video
        cv2.imshow('Press SPACEBAR to capture - ESC to cancel', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACEBAR to capture
            # Save the captured image
            save_path = "data/input/raw_images/webcam_photo.jpg"
            cv2.imwrite(save_path, frame)
            print(f"✅ Photo saved: {save_path}")
            
            # Analyze the photo immediately
            print("🔄 Analyzing hairline...")
            result = detector.analyze_hairline(frame)
            
            if result:
                print("🎉 Analysis successful!")
                print(f"   Hairline Type: {result['hairline_type']}")
                print(f"   Height: {result['hairline_height']:.3f}")
                
                # Show results
                vis_image = detector.visualize_analysis(frame, result)
                cv2.imshow('Hairline Analysis Results', vis_image)
                cv2.waitKey(0)
                
                # Save results
                data_manager.save_analysis_result(result, "webcam_user")
                data_manager.save_visualization(vis_image, "webcam_user")
                print("💾 Results saved to output folders!")
            else:
                print("❌ No face detected. Try again with better lighting.")
            
            break
        elif key == 27:  # ESC to cancel
            print("❌ Capture cancelled")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    take_and_analyze_photo()