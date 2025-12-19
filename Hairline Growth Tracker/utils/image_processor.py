import cv2
import numpy as np

def preprocess_image(image_path):
    """Preprocess image for better analysis"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Resize image for consistent processing
    image = resize_image(image, width=800)
    
    # Enhance contrast
    image = enhance_contrast(image)
    
    return image

def resize_image(image, width=800):
    """Resize image maintaining aspect ratio"""
    height, current_width = image.shape[:2]
    ratio = width / current_width
    new_height = int(height * ratio)
    return cv2.resize(image, (width, new_height))

def enhance_contrast(image):
    """Enhance image contrast for better feature detection"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

def validate_image_quality(image):
    """Validate if image is suitable for analysis"""
    if image is None:
        return False, "Image could not be loaded"
    
    height, width = image.shape[:2]
    
    # Check image size
    if height < 300 or width < 300:
        return False, "Image too small for analysis"
    
    # Check image brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    if brightness < 50:
        return False, "Image too dark"
    elif brightness > 200:
        return False, "Image too bright"
    
    return True, "Image quality OK"