# Bird Detector - Simple Branch

A lightweight bird detection library optimized for AWS Lambda deployment with minimal dependencies.

## Overview

This branch contains the **Simple Detector** implementation - a lightweight anomaly detection system that doesn't require PyTorch or heavy ML dependencies. Perfect for serverless deployments where package size and cold start times are critical.

## Features

- **Lightweight**: No PyTorch dependencies
- **Fast**: Optimized for real-time processing
- **Serverless-ready**: Minimal package size for AWS Lambda
- **Configurable**: YAML-based configuration system

## Installation

This library is designed to be used as a Git submodule. Add it to your project:

```bash
# Add as submodule to your project
git submodule add https://github.com/rayhagimoto/bird-detector.git bird_detector

# Install dependencies
cd bird_detector
pip install -r requirements.txt
```

## Usage

### Basic Detection

```python
from bird_detector.detectors.simple_detector import SimpleDetector
import boto3

# Initialize detector with configuration and S3 client
s3_client = boto3.client('s3')
config = {
    'bucket_name': 'your-s3-bucket',
    'state_folder': 'py',
    'image_size': [64, 64],
    'percentile': 98,
    'alpha': 0.05,
    'min_observations': 25
}

detector = SimpleDetector(config, s3_client)

# Process a PIL Image
from PIL import Image
img = Image.open("path/to/image.jpg")
has_anomaly = detector.predict(img)
print(f"Has anomaly: {has_anomaly}")
```

### Configuration

The detector requires a configuration dictionary and S3 client:

```python
config = {
    'bucket_name': 'your-s3-bucket',      # S3 bucket for state storage
    'state_folder': 'py',                 # Folder within bucket for state files
    'ema_filename': 'ema.npy',           # Filename for exponential moving average
    'scores_filename': 'losses.csv',     # Filename for anomaly scores
    'image_size': [64, 64],              # Target image size [width, height]
    'percentile': 98,                    # Percentile for anomaly threshold
    'alpha': 0.05,                       # EMA smoothing factor
    'min_observations': 25               # Minimum observations before detection
}
```

## API Reference

### SimpleDetector.predict()

Processes a PIL Image and returns whether an anomaly is detected.

**Parameters:**
- `img` (PIL.Image): Input image to analyze

**Returns:**
- `bool`: `True` if anomaly detected, `False` otherwise

**Algorithm:**
1. Resizes image to configured size and normalizes to [0,1]
2. Flattens image and computes exponential moving average (EMA)
3. Calculates anomaly score as log10 of mean squared error
4. Compares score against percentile threshold (after minimum observations)
5. Automatically saves updated EMA and scores to S3

## Architecture

The Simple Detector uses an exponential moving average (EMA) approach:

1. **Image Preprocessing**: Resize to configured size and normalize to [0,1]
2. **Feature Extraction**: Flatten image to 1D array
3. **EMA Computation**: Maintain running average of image features
4. **Anomaly Scoring**: Calculate log10 of mean squared error from EMA
5. **Threshold Detection**: Use percentile-based threshold after minimum observations
6. **State Persistence**: Automatically save EMA and scores to S3

## Dependencies

See `requirements.txt` for the complete list. Core dependencies:
- NumPy - Numerical operations
- PIL (Pillow) - Image processing
- boto3 - AWS S3 client (for state persistence)

## License

MIT License 
