"""
Utility modules for the Hairline Tracker system.

This package contains helper functions and classes for:
- Face detection and landmark extraction
- Image preprocessing and enhancement
- Data validation and quality control
"""

from .face_detector import FaceDetector, create_face_detector, detect_single_face
from .image_processor import preprocess_image, resize_image, enhance_contrast, validate_image_quality

__all__ = [
    'FaceDetector',
    'create_face_detector', 
    'detect_single_face',
    'preprocess_image',
    'resize_image',
    'enhance_contrast',
    'validate_image_quality'
]

# Version information for utils
__version__ = "1.0.0"

def get_utils_info():
    """Return information about the utils package"""
    return {
        'version': __version__,
        'modules': ['face_detector', 'image_processor'],
        'description': 'Utility functions for hairline tracking system'
    }