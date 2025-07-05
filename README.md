# Bird Detector Library

A comprehensive bird detection library with multiple anomaly detection approaches, optimized for research and AWS Lambda deployment.

## Overview

This repository contains a complete bird detection system with two main approaches:
- **Simple Detector**: Lightweight anomaly detection using exponential moving averages (EMA)
- **ConvDetector**: Sophisticated convolutional autoencoder with adaptive learning

The library is designed for both research experimentation and production deployment, with specialized branches for different use cases.

## Repository Structure

```
bird_detector/
├── detectors/                 # Core detection implementations
│   ├── anomaly_detector.py   # Base class for all detectors
│   ├── simple_detector.py    # Lightweight EMA-based detector
│   └── convae_detector.py    # ConvAE-based adaptive detector
├── autoencoder/              # Autoencoder implementations and utilities
│   ├── convolution.py        # Convolutional autoencoder architecture
│   ├── fully_connected.py    # Fully connected autoencoder
│   ├── train.py             # Training utilities
│   ├── scores.py            # Anomaly scoring functions
│   ├── torch_utils.py       # PyTorch utilities (loss, optimizers)
│   ├── otsu_method.py       # Otsu thresholding method
│   ├── plot_utils.py        # Visualization utilities
│   └── interactive_utils.py # Interactive training and evaluation
├── data/                    # Data processing utilities
│   ├── data_loaders.py      # Data loading and batching
│   ├── preprocessing_pipelines.py # Image preprocessing pipelines
│   ├── raw_image_utils.py   # Raw image handling utilities
│   └── image_utils.py       # General image utilities
├── utils.py                 # General utility functions
├── scores.py                # Scoring and evaluation metrics
├── model.py                 # Model definitions and interfaces
├── mlflow_utils.py          # MLflow experiment tracking
└── color_transforms.py      # Color space transformations
```

## Branches

### Main Branch
The main branch contains the complete library with all implementations and utilities. This is the research and development branch with full functionality.

### Simple Branch
Minimal implementation containing only the Simple Detector:
- `detectors/simple_detector.py` - EMA-based anomaly detection
- `detectors/anomaly_detector.py` - Base class
- Minimal dependencies (no PyTorch)
- Optimized for AWS Lambda deployment

### Conv-AE Branch
Minimal implementation containing only the ConvDetector:
- `detectors/convae_detector.py` - ConvAE-based adaptive detection
- `autoencoder/convolution.py` - ConvAE architecture
- `autoencoder/scores.py` - Localized scoring
- `autoencoder/torch_utils.py` - PyTorch utilities
- `autoencoder/otsu_method.py` - Thresholding method
- PyTorch dependencies for deep learning

## Core Components

### Detectors

#### SimpleDetector
- **Approach**: Exponential Moving Average (EMA)
- **Dependencies**: NumPy, PIL, boto3
- **Use Case**: Lightweight, fast deployment
- **Features**: 
  - Minimal memory footprint
  - Fast processing (~50-100ms)
  - No deep learning dependencies
  - S3 state persistence

#### ConvDetector
- **Approach**: Convolutional Autoencoder with Adaptive Learning
- **Dependencies**: PyTorch, torchvision, NumPy, PIL, boto3
- **Use Case**: Sophisticated anomaly detection
- **Features**:
  - Continuous learning on rolling window
  - Burn-in period for model stabilization
  - Dynamic thresholding (98th percentile + Otsu)
  - Localized reconstruction scoring
  - S3 state persistence

### Autoencoder Module

#### Architectures
- **ConvAutoencoder**: Convolutional encoder/decoder with configurable channels
- **SimpleAutoencoder**: Fully connected autoencoder for comparison

#### Utilities
- **Training**: Configurable loss functions and optimizers
- **Scoring**: Localized reconstruction error with Gaussian weighting
- **Visualization**: Plotting and interactive evaluation tools
- **Thresholding**: Otsu method for adaptive thresholds

### Data Processing

#### Pipelines
- **Preprocessing**: Image resizing, normalization, augmentation
- **Loading**: Efficient data loading with batching
- **Raw Image Handling**: Support for various image formats

#### Utilities
- **Color Transforms**: Color space conversions and analysis
- **Image Utils**: General image manipulation functions

## Usage Examples

### Simple Detector
```python
from bird_detector.detectors.simple_detector import SimpleDetector
import boto3

s3_client = boto3.client('s3')
config = {
    'bucket_name': 'your-s3-bucket',
    'image_size': [64, 64],
    'percentile': 98,
    'alpha': 0.05
}

detector = SimpleDetector(config, s3_client)
has_anomaly = detector.predict(image)
```

### ConvDetector
```python
from bird_detector.detectors.convae_detector import ConvDetector
import boto3

s3_client = boto3.client('s3')
config = {
    'bucket_name': 'your-s3-bucket',
    'incubation_period': 200,
    'model': {
        'image_size': 64,
        'latent_dim': 2,
        'enc_channels': [16, 32, 64, 8]
    }
}

detector = ConvDetector(config, s3_client)
has_anomaly = detector.predict(image)
```

## Installation

This library is designed to be used as a Git submodule. Add it to your project:

```bash
# Add as submodule to your project
git submodule add https://github.com/rayhagimoto/bird-detector.git bird_detector

# Install dependencies
cd bird_detector
pip install -r requirements.txt
```

## Development Workflow

1. **Research**: Use main branch for experimentation and development
2. **Deployment**: Use specialized branches (simple/conv-ae) for production
3. **Versioning**: Each branch can be versioned independently
4. **Integration**: Main branch serves as the source of truth for all implementations

## Dependencies

### Main Branch (Full Library)
- PyTorch - Deep learning framework
- torchvision - Image transformations
- NumPy - Numerical operations
- PIL (Pillow) - Image processing
- boto3 - AWS S3 client
- scipy - Scientific computing
- mlflow - Experiment tracking
- matplotlib - Visualization

### Simple Branch
- NumPy - Numerical operations
- PIL (Pillow) - Image processing
- boto3 - AWS S3 client

### Conv-AE Branch
- PyTorch - Deep learning framework
- torchvision - Image transformations
- NumPy - Numerical operations
- PIL (Pillow) - Image processing
- boto3 - AWS S3 client
- scipy - Scientific computing

## License

MIT License 