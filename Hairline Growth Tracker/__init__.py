"""
Hairline Tracker - AI-Powered Telemedicine for Hair Growth Monitoring

A comprehensive system for detecting faces, analyzing hairlines, and tracking
hair growth progress over time using computer vision and machine learning.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .main import HairlineTrackerApp
from .hairline_detector import HairlineDetector
from .progress_tracker import ProgressTracker

__all__ = [
    'HairlineTrackerApp',
    'HairlineDetector', 
    'ProgressTracker'
]

def get_version():
    """Return the current version of the package"""
    return __version__

def about():
    """Print package information"""
    print(f"Hairline Tracker v{__version__}")
    print("AI-powered telemedicine system for hair growth monitoring")
    print("Features:")
    print("- Face detection and landmark analysis")
    print("- Hairline detection and measurement") 
    print("- Progress tracking over time")
    print("- Comprehensive reporting and visualization")