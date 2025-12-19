import os
import json
import cv2
import shutil
from datetime import datetime
import numpy as np

class DataManager:
    def __init__(self):
        self.setup_directories()
        self.dataset_links = {
            'celeba': 'http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html',
            'wider_face': 'http://shuoyang1213.me/WIDERFACE/',
        }
        print("✅ Data Manager initialized successfully!")
    
    def setup_directories(self):
        """Create all necessary data directories"""
        directories = [
            'data/input/raw_images',
            'data/input/processed_images', 
            'data/input/user_data',
            'data/input/datasets',
            'data/output/analysis_results',
            'data/output/progress_reports',
            'data/output/visualizations',
            'data/output/exports'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("✅ All data directories created successfully!")
    
    def create_sample_images(self):
        """Create sample synthetic images for testing"""
        print("🖼️ Creating sample images for testing...")
        
        # Create 5 sample images with different hairline positions
        for i in range(5):
            # Create a blank image
            img = np.ones((500, 500, 3), dtype=np.uint8) * 255  # White background
            
            # Draw a simple face with varying hairline
            hairline_y = 100 + (i * 15)  # Varying hairline position
            
            # Draw hair (brown)
            cv2.rectangle(img, (150, hairline_y), (350, 200), (101, 67, 33), -1)
            
            # Draw forehead (skin tone)
            cv2.rectangle(img, (150, 200), (350, 300), (255, 229, 204), -1)
            
            # Draw eyes
            cv2.circle(img, (200, 250), 10, (0, 0, 0), -1)  # Left eye
            cv2.circle(img, (300, 250), 10, (0, 0, 0), -1)  # Right eye
            
            # Draw mouth
            cv2.ellipse(img, (250, 320), (30, 10), 0, 0, 180, (0, 0, 0), 2)
            
            # Save sample image
            filename = f"data/input/raw_images/sample_{i+1:02d}.jpg"
            cv2.imwrite(filename, img)
            
        print(f"✅ Created 5 sample images in data/input/raw_images/")
    
    def validate_image(self, image_path):
        """Validate if image is suitable for analysis"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Cannot read image file"
            
            height, width = img.shape[:2]
            if height < 300 or width < 300:
                return False, "Image too small (min 300x300 required)"
            
            # Check if image is too dark or too bright
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            if avg_brightness < 50:
                return False, "Image too dark"
            elif avg_brightness > 200:
                return False, "Image too bright"
                
            return True, "Image validated successfully"
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
    
    def save_input_image(self, image, user_id="default_user", image_name=None):
        """Save input image with proper naming"""
        if image_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = f"{user_id}_{timestamp}.jpg"
        
        input_path = f"data/input/raw_images/{image_name}"
        success = cv2.imwrite(input_path, image)
        
        if success:
            print(f"💾 Input image saved: {input_path}")
            return input_path
        else:
            print(f"❌ Failed to save image: {input_path}")
            return None
    
    def save_processed_image(self, image, user_id, description="processed"):
        """Save processed image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/input/processed_images/{user_id}_{timestamp}_{description}.jpg"
        
        success = cv2.imwrite(output_path, image)
        if success:
            print(f"💾 Processed image saved: {output_path}")
            return output_path
        else:
            print(f"❌ Failed to save processed image: {output_path}")
            return None
    
    def save_analysis_result(self, result, user_id, timestamp=None):
        """Save analysis results as JSON"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_path = f"data/output/analysis_results/{user_id}_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.datetime64):
                return str(obj)
            return obj
        
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=convert_numpy_types)
        
        print(f"💾 Analysis results saved: {result_path}")
        return result_path
    
    def save_progress_report(self, report, user_id, report_type="progress"):
        """Save progress report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"data/output/progress_reports/{user_id}_{timestamp}_{report_type}.txt"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"💾 Progress report saved: {report_path}")
        return report_path
    
    def save_visualization(self, image, user_id, viz_type="analysis"):
        """Save visualization image"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = f"data/output/visualizations/{user_id}_{timestamp}_{viz_type}.jpg"
        
        success = cv2.imwrite(viz_path, image)
        if success:
            print(f"💾 Visualization saved: {viz_path}")
            return viz_path
        else:
            print(f"❌ Failed to save visualization: {viz_path}")
            return None
    
    def get_user_history(self, user_id):
        """Get analysis history for a user"""
        analysis_files = []
        results_path = "data/output/analysis_results"
        
        if not os.path.exists(results_path):
            return []
            
        for filename in os.listdir(results_path):
            if filename.startswith(user_id) and filename.endswith('.json'):
                analysis_files.append({
                    'filename': filename,
                    'path': os.path.join(results_path, filename),
                    'timestamp': filename.replace(f"{user_id}_", "").replace(".json", "")
                })
        
        return sorted(analysis_files, key=lambda x: x['timestamp'])
    
    def export_user_data(self, user_id, export_format='json'):
        """Export all user data for backup or transfer"""
        export_data = {
            'user_id': user_id,
            'export_date': datetime.now().isoformat(),
            'analyses': [],
            'reports': []
        }
        
        # Collect analysis results
        analysis_files = self.get_user_history(user_id)
        for analysis_file in analysis_files:
            try:
                with open(analysis_file['path'], 'r') as f:
                    analysis_data = json.load(f)
                    export_data['analyses'].append(analysis_data)
            except Exception as e:
                print(f"⚠️ Could not load analysis file {analysis_file['path']}: {e}")
        
        # Export to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"data/output/exports/{user_id}_{timestamp}.{export_format}"
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📤 User data exported: {export_path}")
        return export_path
    
    def batch_process_images(self, input_folder="data/input/raw_images", user_id="batch_user"):
        """Process all images in a folder"""
        valid_images = []
        invalid_images = []
        
        if not os.path.exists(input_folder):
            print(f"❌ Input folder not found: {input_folder}")
            return [], []
        
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(input_folder, filename)
                
                is_valid, message = self.validate_image(image_path)
                if is_valid:
                    valid_images.append(image_path)
                else:
                    invalid_images.append((filename, message))
        
        print(f"📊 Found {len(valid_images)} valid images and {len(invalid_images)} invalid images")
        
        if invalid_images:
            print("❌ Invalid images:")
            for img, msg in invalid_images:
                print(f"   - {img}: {msg}")
        
        return valid_images, invalid_images
    
    def cleanup_user_data(self, user_id, days_old=30):
        """Clean up old user data"""
        import time
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        cleaned_files = 0
        
        # Clean various directories
        directories_to_clean = [
            'data/input/raw_images',
            'data/input/processed_images',
            'data/output/analysis_results',
            'data/output/progress_reports',
            'data/output/visualizations'
        ]
        
        for directory in directories_to_clean:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    if filename.startswith(user_id):
                        filepath = os.path.join(directory, filename)
                        if os.path.isfile(filepath) and os.path.getctime(filepath) < cutoff_time:
                            os.remove(filepath)
                            cleaned_files += 1
        
        print(f"🧹 Cleaned up {cleaned_files} old files for user {user_id}")
        return cleaned_files

# Test function
def test_data_manager():
    """Test the data manager functionality"""
    dm = DataManager()
    print("🧪 Testing Data Manager...")
    
    # Test directory creation
    assert os.path.exists('data/input/raw_images'), "Raw images directory not created"
    assert os.path.exists('data/output/analysis_results'), "Analysis results directory not created"
    
    # Test creating sample images
    dm.create_sample_images()
    
    # Test saving a sample image
    sample_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    saved_path = dm.save_input_image(sample_image, "test_user")
    assert saved_path is not None, "Failed to save test image"
    
    print("✅ Data Manager test completed successfully!")

if __name__ == "__main__":
    test_data_manager()