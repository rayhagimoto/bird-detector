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
    SimpleAutoencoderDetector,
    train_model,
    get_loss_function,
    get_optimizer,
)

# Data utilities
from .data.data_loaders import SimpleBirdDataset, MultiSetBirdDataset

# Utility functions
from .utils import *
from .scores import *
from .mlflow_utils import *

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
