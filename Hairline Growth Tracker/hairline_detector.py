import cv2
import numpy as np
from sklearn.cluster import KMeans
from utils.face_detector import FaceDetector

class HairlineDetector:
    def __init__(self):
        """
        Initialize Hairline Detector
        """
        self.face_detector = FaceDetector()
        
    def analyze_hairline(self, image):
        """
        Main function to analyze hairline from image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            dict: Hairline analysis results including metrics and points
        """
        try:
            # Detect face and landmarks
            detection_result = self.face_detector.detect_face(image)
            
            if not detection_result or not detection_result['success']:
                print("No face detected in the image")
                return None
            
            # Extract landmarks and regions
            landmarks = detection_result['landmarks']
            forehead_region = self.face_detector.get_forehead_region(landmarks)
            
            # Detect hairline points
            hairline_points = self.detect_hairline_points(image, landmarks, forehead_region)
            
            # Calculate comprehensive metrics
            metrics = self.calculate_metrics(landmarks, hairline_points, image.shape)
            
            # Determine hairline classification
            hairline_type = self.classify_hairline(metrics)
            
            return {
                'face_landmarks': landmarks,
                'hairline_points': hairline_points,
                'forehead_region': forehead_region,
                'hairline_height': metrics['hairline_height'],
                'forehead_ratio': metrics['forehead_ratio'],
                'density_score': metrics['density_score'],
                'symmetry_score': metrics['symmetry_score'],
                'recession_score': metrics['recession_score'],
                'hairline_type': hairline_type,
                'analysis_quality': metrics['analysis_quality']
            }
            
        except Exception as e:
            print(f"Error in hairline analysis: {e}")
            return None
    
    def detect_hairline_points(self, image, landmarks, forehead_region):
        """
        Detect hairline points using edge detection
        """
        if forehead_region is None:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create mask for forehead region
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        pts = forehead_region.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply mask
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Find contours
        contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hairline_points = []
        for contour in contours:
            for point in contour:
                x, y = point[0]
                hairline_points.append([x, y])
        
        return np.array(hairline_points)
    
    def calculate_metrics(self, landmarks, hairline_points, image_shape):
        """
        Calculate hairline metrics
        """
        height, width = image_shape[:2]
        
        # Basic hairline height
        if len(hairline_points) > 0:
            hairline_y = np.min(hairline_points[:, 1])
            hairline_height = hairline_y / height
        else:
            # Fallback: use forehead landmarks
            forehead_y = np.min([landmarks[i][1] for i in [10, 67, 69, 104, 108, 109, 151, 337, 338, 297] 
                               if i < len(landmarks)])
            hairline_y = forehead_y
            hairline_height = forehead_y / height
        
        # Forehead ratio
        eyebrow_y = np.mean([landmarks[i][1] for i in [105, 334, 336] if i < len(landmarks)])
        chin_y = np.max([landmarks[i][1] for i in [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234] 
                        if i < len(landmarks)])
        
        face_height = chin_y - hairline_y
        forehead_height = eyebrow_y - hairline_y
        
        if face_height > 0:
            forehead_ratio = forehead_height / face_height
        else:
            forehead_ratio = 0.3
        
        # Density score
        if len(hairline_points) > 0:
            density_score = min(len(hairline_points) / 50, 1.0)
        else:
            density_score = 0.3
        
        # Symmetry score
        symmetry_score = self.calculate_symmetry(hairline_points, width)
        
        # Recession score (simplified)
        recession_score = 0.3  # Default moderate
        
        # Analysis quality
        analysis_quality = 0.7  # Default good
        
        return {
            'hairline_height': hairline_height,
            'forehead_ratio': forehead_ratio,
            'density_score': density_score,
            'symmetry_score': symmetry_score,
            'recession_score': recession_score,
            'analysis_quality': analysis_quality
        }
    
    def calculate_symmetry(self, hairline_points, image_width):
        """
        Calculate hairline symmetry
        """
        if len(hairline_points) < 2:
            return 0.5
        
        midpoint = image_width / 2
        
        # Split points into left and right
        left_points = [p for p in hairline_points if p[0] < midpoint]
        right_points = [p for p in hairline_points if p[0] > midpoint]
        
        if len(left_points) == 0 or len(right_points) == 0:
            return 0.5
        
        # Mirror right points
        mirrored_right = [[image_width - p[0], p[1]] for p in right_points]
        
        # Calculate height symmetry
        left_avg_y = np.mean([p[1] for p in left_points])
        right_avg_y = np.mean([p[1] for p in mirrored_right])
        symmetry_score = 1.0 - min(abs(left_avg_y - right_avg_y) / 50, 1.0)
        
        return symmetry_score
    
    def classify_hairline(self, metrics):
        """
        Classify hairline type based on metrics
        """
        height = metrics['hairline_height']
        recession = metrics['recession_score']
        symmetry = metrics['symmetry_score']
        
        if recession > 0.7:
            return "Receding"
        elif recession > 0.5:
            return "Mature"
        elif height < 0.15:
            return "Low"
        elif height > 0.25:
            return "High"
        elif symmetry < 0.6:
            return "Asymmetric"
        else:
            return "Normal"
    
    def visualize_analysis(self, image, analysis_result, save_path=None):
        """
        Visualize hairline analysis results
        """
        vis_image = image.copy()
        
        if not analysis_result:
            return vis_image
        
        # Draw face landmarks
        for landmark in analysis_result['face_landmarks']:
            x, y = int(landmark[0]), int(landmark[1])
            cv2.circle(vis_image, (x, y), 2, (0, 255, 0), -1)
        
        # Draw hairline points
        for point in analysis_result['hairline_points']:
            x, y = int(point[0]), int(point[1])
            cv2.circle(vis_image, (x, y), 3, (255, 0, 0), -1)
        
        # Add text information
        y_offset = 30
        line_height = 25
        
        info_texts = [
            f"Hairline Type: {analysis_result['hairline_type']}",
            f"Hairline Height: {analysis_result['hairline_height']:.3f}",
            f"Forehead Ratio: {analysis_result['forehead_ratio']:.3f}",
            f"Density Score: {analysis_result['density_score']:.3f}",
            f"Symmetry Score: {analysis_result['symmetry_score']:.3f}",
        ]
        
        for i, text in enumerate(info_texts):
            y_pos = y_offset + i * line_height
            cv2.putText(vis_image, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image

# Utility function for quick analysis
def analyze_single_image(image_path, visualize=True):
    """
    Convenience function for single image analysis
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Create detector and analyze
    detector = HairlineDetector()
    result = detector.analyze_hairline(image)
    
    if result and visualize:
        # Display results
        vis_image = detector.visualize_analysis(image, result)
        cv2.imshow('Hairline Analysis', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result

# Example usage
if __name__ == "__main__":
    # Test the hairline detector
    image_path = "data/input/raw_images/sample_01.jpg"
    
    result = analyze_single_image(image_path)
    
    if result:
        print("Hairline Analysis Results:")
        print(f"Type: {result['hairline_type']}")
        print(f"Height: {result['hairline_height']:.3f}")
        print(f"Forehead Ratio: {result['forehead_ratio']:.3f}")
        print(f"Density: {result['density_score']:.3f}")
        print(f"Symmetry: {result['symmetry_score']:.3f}")
    else:
        print("Analysis failed.")