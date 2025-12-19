"""
Data Management Package for Hairline Tracker

Handles all data input/output operations including:
- Image storage and retrieval
- Analysis result saving
- Progress reporting
- Data export functionality
"""

from .data_manager import DataManager

__all__ = ['DataManager']

__version__ = "1.0.0"

def about():
    """Print data package information"""
    print("Data Management Package for Hairline Tracker")
    print("Manages all input/output operations and file organization")