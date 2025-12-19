"""
Model loading and management utilities
"""

import os
import cv2
import json

class ModelLoader:
    def __init__(self):
        self.loaded_models = {}
    
    def ensure_model_directories(self):
        """Create necessary directories for models"""
        os.makedirs('models/trained_models', exist_ok=True)
    
    def save_model_metadata(self, model_name, metadata):
        """Save model metadata"""
        metadata_path = f'models/trained_models/{model_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model_metadata(self, model_name):
        """Load model metadata"""
        metadata_path = f'models/trained_models/{model_name}_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None