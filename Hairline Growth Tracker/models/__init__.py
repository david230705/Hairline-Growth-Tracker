"""
Models package for Hairline Tracker

Contains model loading, saving, and configuration utilities
for face detection and hairline analysis.
"""

from .model_loader import ModelLoader
from .config import ModelConfig

__all__ = ['ModelLoader', 'ModelConfig']

__version__ = "1.0.0"

def get_models_info():
    """Return information about the models package"""
    return {
        'version': __version__,
        'description': 'Model management for hairline tracking system',
        'modules': ['model_loader', 'config']
    }