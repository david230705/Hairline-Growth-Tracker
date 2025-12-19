import cv2
import numpy as np
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        """
        Initialize Face Detector using MediaPipe Face Mesh
        """
        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Define important facial landmarks for hairline analysis
        self.landmark_indices = {
            'forehead': [10, 67, 69, 104, 108, 109, 151, 337, 338, 297],
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        }
    
    def detect_face(self, image):
        """
        Detect face and landmarks in the image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            dict: Contains detection results and landmarks
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Extract landmarks for the first face
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = self.extract_landmark_coordinates(face_landmarks, width, height)
        
        # Get face bounding box
        bbox = self.get_face_bounding_box(landmarks, width, height)
        
        return {
            'landmarks': landmarks,
            'bbox': bbox,
            'success': True
        }
    
    def extract_landmark_coordinates(self, face_landmarks, image_width, image_height):
        """
        Extract landmark coordinates and convert to pixel values
        """
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            landmarks.append([x, y])
        
        return landmarks
    
    def get_face_bounding_box(self, landmarks, image_width, image_height):
        """
        Calculate bounding box around the face
        """
        if not landmarks:
            return None
        
        landmarks_array = np.array(landmarks)
        x_min = np.min(landmarks_array[:, 0])
        y_min = np.min(landmarks_array[:, 1])
        x_max = np.max(landmarks_array[:, 0])
        y_max = np.max(landmarks_array[:, 1])
        
        # Add some padding
        padding_x = int((x_max - x_min) * 0.05)
        padding_y = int((y_max - y_min) * 0.05)
        
        # Ensure coordinates are within image boundaries
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(image_width, x_max + padding_x)
        y_max = min(image_height, y_max + padding_y)
        
        return {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min
        }
    
    def get_forehead_region(self, landmarks):
        """
        Get extended forehead region for hairline analysis
        """
        # Use forehead landmarks as base
        forehead_indices = [10, 67, 69, 104, 108, 109, 151, 337, 338, 297]
        forehead_base = []
        for idx in forehead_indices:
            if idx < len(landmarks):
                forehead_base.append(landmarks[idx])
        
        if not forehead_base:
            return None
        
        forehead_array = np.array(forehead_base)
        
        # Extend the forehead region upward for hairline detection
        y_min = np.min(forehead_array[:, 1])
        y_max = np.max(forehead_array[:, 1])
        x_min = np.min(forehead_array[:, 0])
        x_max = np.max(forehead_array[:, 0])
        
        # Create extended region (above the detected forehead)
        extension = (y_max - y_min) * 0.5  # Extend 50% upward
        
        extended_forehead = [
            [x_min, max(0, y_min - extension)],
            [x_max, max(0, y_min - extension)],
            [x_max, y_max],
            [x_min, y_max]
        ]
        
        return np.array(extended_forehead)
    
    def is_frontal_face(self, landmarks, threshold=0.85):
        """
        Check if face is frontal (facing forward) for accurate hairline analysis
        """
        if len(landmarks) < 468:  # MediaPipe Face Mesh has 468 landmarks
            return False
        
        # Check symmetry using key points
        left_face = [landmarks[i] for i in [33, 133, 362]]  # Left eye, left face
        right_face = [landmarks[i] for i in [263, 361, 130]]  # Right eye, right face
        
        left_array = np.array(left_face)
        right_array = np.array(right_face)
        
        # Calculate symmetry score
        left_center = np.mean(left_array, axis=0)
        right_center = np.mean(right_array, axis=0)
        
        image_center = np.array([landmarks[1][0], landmarks[1][1]])  # Use nose base as reference
        
        left_dist = np.linalg.norm(left_center - image_center)
        right_dist = np.linalg.norm(right_center - image_center)
        
        symmetry_score = min(left_dist, right_dist) / max(left_dist, right_dist)
        
        return symmetry_score >= threshold
    
    def release(self):
        """Release resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

# Utility functions for face detection
def create_face_detector():
    """Factory function to create face detector"""
    return FaceDetector()

def detect_single_face(image_path):
    """
    Convenience function to detect face in a single image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Detection results
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Create detector and detect face
    detector = FaceDetector()
    result = detector.detect_face(image)
    
    # Release resources
    detector.release()
    
    return result

# Example usage and testing
if __name__ == "__main__":
    # Test the face detector
    detector = FaceDetector()
    
    # Create a sample image for testing
    sample_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
    cv2.rectangle(sample_image, (150, 150), (350, 350), (0, 0, 0), 2)
    
    # Detect face
    result = detector.detect_face(sample_image)
    
    if result and result['success']:
        print("Face detection test completed!")
        print(f"Number of landmarks: {len(result['landmarks'])}")
    else:
        print("No face detected in sample image.")
    
    # Release resources
    detector.release()