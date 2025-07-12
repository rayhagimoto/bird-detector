"""
Anomaly detection implementations for bird detection.

This module provides various anomaly detection approaches:
- SimpleDetector: EMA-based lightweight detection
- ConvDetector: ConvAE-based sophisticated detection  
- OpenCVDetector: OpenCV-based detection
"""

from .anomaly_detector import AnomalyDetector
from .convae_detector import ConvDetector
from .opencv_detector import OpenCVDetector

__all__ = [
    "AnomalyDetector",
    "ConvDetector",
    "OpenCVDetector",
]
