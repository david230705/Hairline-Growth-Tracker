import cv2
import os
import sys
from datetime import datetime
from hairline_detector import HairlineDetector
from progress_tracker import ProgressTracker
from data.data_manager import DataManager

class HairlineTrackerApp:
    def __init__(self):
        self.detector = HairlineDetector()
        self.tracker = ProgressTracker()
        self.data_manager = DataManager()
        print("🚀 Hairline Tracker initialized successfully!")
    
    def setup_environment(self):
        """Setup the complete environment"""
        print("🔧 Setting up environment...")
        self.data_manager.setup_directories()
        self.data_manager.create_sample_images()
        print("✅ Environment setup complete!")
    
    def take_photo(self):
        """Take photo with webcam and analyze immediately"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot access webcam. Make sure it's connected and not used by another application.")
            return
        
        print("📷 Webcam activated!")
        print("💡 Instructions:")
        print("   - Look straight at the camera")
        print("   - Make sure your face is well-lit")
        print("   - Press SPACEBAR to capture photo")
        print("   - Press ESC to cancel")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture image from webcam")
                break
            
            # Display the live video
            cv2.imshow('Press SPACEBAR to capture - ESC to cancel', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # SPACEBAR to capture
                user_id = input("Enter user ID for this photo: ").strip() or "webcam_user"
                
                # Save the captured image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"data/input/raw_images/{user_id}_webcam_{timestamp}.jpg"
                success = cv2.imwrite(save_path, frame)
                
                if success:
                    print(f"✅ Photo saved: {save_path}")
                    
                    # Analyze the photo immediately
                    print("🔄 Analyzing hairline...")
                    result = self.detector.analyze_hairline(frame)
                    
                    if result:
                        print("🎉 Analysis successful!")
                        print(f"   Hairline Type: {result['hairline_type']}")
                        print(f"   Height: {result['hairline_height']:.3f}")
                        print(f"   Density: {result['density_score']:.3f}")
                        
                        # Save results
                        self.tracker.save_analysis(user_id, timestamp, result)
                        self.data_manager.save_analysis_result(result, user_id, timestamp)
                        
                        # Show and save visualization
                        vis_image = self.detector.visualize_analysis(frame, result)
                        self.data_manager.save_visualization(vis_image, user_id)
                        cv2.imshow('Hairline Analysis Results', vis_image)
                        print("📊 Press any key to close results...")
                        cv2.waitKey(0)
                        
                        print("💾 All results saved to output folders!")
                    else:
                        print("❌ No face detected in the photo.")
                        print("   Please ensure:")
                        print("   - Your face is clearly visible")
                        print("   - Good lighting conditions")
                        print("   - Front-facing pose")
                else:
                    print("❌ Failed to save photo")
                
                break
            elif key == 27:  # ESC to cancel
                print("❌ Capture cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_single_image(self, image_path=None, user_id=None):
        """Process a single image and analyze hairline"""
        if image_path is None:
            image_path = input("Enter image path: ").strip()
        
        if user_id is None:
            user_id = input("Enter user ID (default: 'user1'): ").strip() or "user1"
        
        print(f"📷 Processing image: {image_path}")
        
        # Validate image
        is_valid, message = self.data_manager.validate_image(image_path)
        if not is_valid:
            print(f"❌ Image validation failed: {message}")
            return None
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Save input image
        saved_path = self.data_manager.save_input_image(image, user_id)
        
        # Detect hairline and get metrics
        print("🔍 Analyzing hairline...")
        result = self.detector.analyze_hairline(image)
        
        if result:
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.tracker.save_analysis(user_id, timestamp, result)
            
            # Save analysis result
            analysis_path = self.data_manager.save_analysis_result(result, user_id, timestamp)
            
            # Create and save visualization
            vis_image = self.detector.visualize_analysis(image, result)
            vis_path = self.data_manager.save_visualization(vis_image, user_id)
            
            print("✅ Analysis completed successfully!")
            print(f"📊 Hairline Type: {result['hairline_type']}")
            print(f"📏 Height: {result['hairline_height']:.3f}")
            print(f"📈 Density: {result['density_score']:.3f}")
            print(f"⚖️  Symmetry: {result['symmetry_score']:.3f}")
            
            # Display results
            cv2.imshow('Hairline Analysis Results', vis_image)
            print("🖼️ Press any key to close the visualization window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return result
        else:
            print("❌ Hairline analysis failed - no face detected")
            return None
    
    def process_batch_images(self, input_folder=None, user_id=None):
        """Process all images in a folder"""
        if input_folder is None:
            input_folder = input("Enter folder path (default: 'data/input/raw_images'): ").strip() or "data/input/raw_images"
        
        if user_id is None:
            user_id = input("Enter user ID (default: 'batch_user'): ").strip() or "batch_user"
        
        print(f"🔄 Processing batch images from: {input_folder}")
        
        valid_images, invalid_images = self.data_manager.batch_process_images(input_folder, user_id)
        
        if not valid_images:
            print("❌ No valid images found to process")
            return []
        
        results = []
        for image_path in valid_images:
            print(f"🔍 Processing: {os.path.basename(image_path)}")
            result = self.process_single_image(image_path, user_id)
            if result:
                results.append(result)
        
        print(f"\n📊 Batch processing complete!")
        print(f"✅ Successful analyses: {len(results)}")
        print(f"❌ Failed analyses: {len(valid_images) - len(results)}")
        
        return results
    
    def track_progress(self, user_id=None):
        """Generate progress report for a user"""
        if user_id is None:
            user_id = input("Enter user ID (default: 'user1'): ").strip() or "user1"
        
        print(f"📈 Generating progress report for: {user_id}")
        
        report = self.tracker.generate_report(user_id)
        
        if report:
            # Save progress report
            report_path = self.data_manager.save_progress_report(report, user_id)
            print(f"✅ Progress report generated: {report_path}")
            
            # Export user data
            export_path = self.data_manager.export_user_data(user_id)
            print(f"📤 User data exported: {export_path}")
            
            return report
        else:
            print("❌ No data available for progress tracking")
            return None
    
    def show_user_history(self, user_id=None):
        """Show user's analysis history"""
        if user_id is None:
            user_id = input("Enter user ID (default: 'user1'): ").strip() or "user1"
        
        print(f"📋 Analysis history for: {user_id}")
        
        history = self.data_manager.get_user_history(user_id)
        
        if not history:
            print("   No analysis history found")
            return
        
        print(f"   Found {len(history)} analyses:")
        for i, analysis in enumerate(history, 1):
            print(f"   {i}. {analysis['timestamp']} - {analysis['filename']}")
    
    def download_sample_datasets(self):
        """Download sample datasets for training"""
        print("📥 Available Datasets:")
        print("1. CelebA - Large-scale Face Attributes")
        print("2. WIDER FACE - Face Detection Benchmark") 
        print("3. Create Sample Images (Recommended for testing)")
        
        dataset_choice = input("Choose dataset (1-3): ").strip()
        if dataset_choice == '1':
            print("📥 CelebA dataset: Visit http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
            print("   Download 'Align&Cropped Images' and extract to data/input/datasets/celeba/")
        elif dataset_choice == '2':
            print("📥 WIDER FACE dataset: Visit http://shuoyang1213.me/WIDERFACE/")
            print("   Download WIDER_train.zip and extract to data/input/datasets/wider_face/")
        else:
            self.data_manager.create_sample_images()

def main():
    """Main function with menu interface"""
    app = HairlineTrackerApp()
    
    while True:
        print("\n" + "="*50)
        print("🧬 HAIRLINE TRACKER - AI TELEMEDICINE")
        print("="*50)
        print("1. Setup Environment (First Time)")
        print("2. Take Photo with Webcam")
        print("3. Process Single Image")
        print("4. Process Batch Images") 
        print("5. Track Progress")
        print("6. Show History")
        print("7. Download Sample Datasets")
        print("8. Exit")
        print("-"*50)
        
        choice = input("Enter your choice (1-8): ").strip()
        
        if choice == '1':
            app.setup_environment()
            
        elif choice == '2':
            app.take_photo()
            
        elif choice == '3':
            app.process_single_image()
            
        elif choice == '4':
            app.process_batch_images()
            
        elif choice == '5':
            report = app.track_progress()
            if report:
                print("\n" + "="*50)
                print(report)
                print("="*50)
            
        elif choice == '6':
            app.show_user_history()
            
        elif choice == '7':
            app.download_sample_datasets()
            
        elif choice == '8':
            print("👋 Thank you for using Hairline Tracker!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()