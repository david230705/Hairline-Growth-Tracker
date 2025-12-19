"""
Model configuration settings
"""

class ModelConfig:
    # Face detection settings
    FACE_DETECTION_CONFIDENCE = 0.5
    FACE_MODEL_SELECTION = 0
    
    # Hairline detection settings
    HAIRLINE_CONFIDENCE_THRESHOLD = 0.6
    MAX_FACES = 1
    
    # Model paths (for future trained models)
    MODEL_PATHS = {
        'face_detector': 'models/trained_models/face_detector.pb',
        'hairline_segmentor': 'models/trained_models/hairline_model.h5'
    }