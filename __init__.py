"""
Bird Detector Library

A comprehensive bird detection library with multiple anomaly detection approaches,
optimized for research and AWS Lambda deployment.

Main components:
- SimpleDetector: Lightweight EMA-based anomaly detection
- ConvDetector: Sophisticated convolutional autoencoder with adaptive learning
- Autoencoder utilities for training and evaluation
- Data processing pipelines for image handling
"""

__version__ = "0.1.0"
__author__ = "Ray Hagimoto"

# Core detectors
from .detectors.simple_detector import SimpleDetector
from .detectors.convae_detector import ConvDetector
from .detectors.opencv_detector import OpenCVDetector

# Autoencoder models and utilities
from .autoencoder import (
    ConvAutoencoder,
    SimpleAutoencoder,
    train_model,
    get_loss_function,
    get_optimizer,
)

# Data utilities
from .data import SimpleBirdDataset, MultiSetBirdDataset

# Expose package-level utilities from utils subpackage
from .utils.mlflow_utils import *
from .utils.scores import *
from .utils.color_transforms import *

__all__ = [
    # Detectors
    "SimpleDetector",
    "ConvDetector", 
    "OpenCVDetector",
    
    # Autoencoder models
    "ConvAutoencoder",
    "SimpleAutoencoder",
    "SimpleAutoencoderDetector",
    
    # Training utilities
    "train_model",
    "get_loss_function",
    "get_optimizer",
    
    # Data utilities
    "SimpleBirdDataset",
    "MultiSetBirdDataset",
]
