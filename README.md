# Bird Detector - Conv-AE Branch

A convolutional autoencoder-based bird detection library optimized for AWS Lambda deployment with adaptive learning capabilities.

## Overview

This branch contains the **ConvDetector** implementation - a sophisticated anomaly detection system using convolutional autoencoders with rolling window adaptation. The detector continuously learns from incoming images and maintains state in S3 for persistent operation across Lambda invocations.

## Features

- **Adaptive Learning**: Continuously trains on rolling window of recent images
- **State Persistence**: Maintains model weights, image window, and scores in S3
- **Burn-in Period**: Initial incubation period for model stabilization
- **Dynamic Thresholding**: Uses 98th percentile and Otsu method for adaptive thresholds
- **Lambda Optimized**: Memory-efficient with configurable batch sizes and window limits
- **Localized Scoring**: Uses Gaussian-weighted reconstruction error for precise anomaly detection

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Detection

```python
from detectors.convae_detector import ConvDetector
import boto3
from PIL import Image

# Initialize detector with configuration and S3 client
s3_client = boto3.client('s3')
config = {
    'bucket_name': 'your-s3-bucket',
    'state_folder': 'py',
    'incubation_period': 200,
    'model': {
        'image_size': 64,
        'latent_dim': 2,
        'enc_channels': [16, 32, 64, 8],
        'dec_channels': [64, 32, 16, 3]
    }
}

detector = ConvDetector(config, s3_client)

# Process a PIL Image
img = Image.open("path/to/image.jpg")
has_anomaly = detector.predict(img)
print(f"Has anomaly: {has_anomaly}")
```

### Configuration

The detector requires a comprehensive configuration dictionary:

```python
config = {
    # S3 Configuration
    'bucket_name': 'your-s3-bucket',           # S3 bucket for state storage
    'state_folder': 'py',                      # Folder within bucket for state files
    'scores_key': 'scores.npy',                # Filename for anomaly scores
    'weights_key': 'model_weights.pth',        # Filename for model weights
    'img_window_key': 'img_window.npy',        # Filename for image window
    
    # Training Configuration
    'incubation_period': 200,                  # Images to process before detection starts
    'incubation_steps': 5,                     # Training steps during incubation
    'incubation_lr': 1e-4,                     # Learning rate during incubation
    'steps_per_image': 2,                      # Training steps per new image
    'lr_per_image': 1e-6,                      # Learning rate per image
    'enable_training': True,                   # Enable/disable online training
    
    # Lambda Optimizations
    'max_window_size': 200,                    # Maximum images in rolling window
    'min_batch_size': 32,                      # Minimum batch size for training
    
    # Model Architecture
    'model': {
        'image_size': 64,                      # Input image size (int or tuple)
        'latent_dim': 2,                       # Latent space dimension
        'enc_channels': [16, 32, 64, 8],      # Encoder channel sequence
        'enc_kernel_sizes': [3, 3, 3, 3],     # Encoder kernel sizes
        'dec_channels': [64, 32, 16, 3],      # Decoder channel sequence
        'dec_kernel_sizes': [3, 3, 3, 3],     # Decoder kernel sizes
        'pool_kernel': 2,                      # Pooling kernel size
        'upsample_mode': 'nearest'             # Upsampling mode
    },
    
    # Loss Function
    'loss_fn': 'log_mse'                       # Loss function type
}
```

## API Reference

### ConvDetector.predict()

Processes a PIL Image and returns whether an anomaly is detected.

**Parameters:**
- `img` (PIL.Image): Input image to analyze

**Returns:**
- `bool`: `True` if anomaly detected, `False` otherwise

**Algorithm Flow:**
1. **Image Transformation**: Resize to model input size and convert to tensor
2. **State Loading**: Load current image window, scores, and image count from S3
3. **Burn-in Period**: For first `incubation_period` images:
   - Always return `False`
   - Add image to window and calculate score
   - Train model if window size ≥ `min_batch_size`
   - Save updated state to S3
4. **Normal Prediction**: After burn-in period:
   - Add image to rolling window (respecting `max_window_size`)
   - Calculate anomaly score using localized reconstruction error
   - Train model on current window if enabled
   - Calculate dynamic threshold using 98th percentile and Otsu method
   - Save updated state to S3
   - Return `anomaly_score > threshold`

### ConvDetector.get_anomaly_score()

Get anomaly score for an image without updating state.

**Parameters:**
- `img` (PIL.Image): Input image to analyze

**Returns:**
- `float`: Anomaly score (log-transformed reconstruction error)

### ConvDetector.get_current_state()

Get current state information for monitoring.

**Returns:**
```python
{
    'window_size': int,                    # Current number of images in window
    'total_images_processed': int,         # Total images processed
    'incubation_period': int,              # Configured incubation period
    'in_burn_in_period': bool,             # Whether still in burn-in
    'total_scores': int,                   # Number of anomaly scores
    'current_threshold': float,            # Current dynamic threshold
    'recent_scores': list,                 # Last 10 anomaly scores
    'model_device': str,                   # Device (cpu/cuda)
    'training_enabled': bool               # Whether training is enabled
}
```

## Architecture

The ConvDetector uses an adaptive approach:

1. **Convolutional Autoencoder**: Encodes images to latent space and reconstructs them
2. **Localized Scoring**: Uses Gaussian-weighted reconstruction error for precise anomaly detection
3. **Rolling Window**: Maintains recent images for continuous learning
4. **Adaptive Training**: Freezes encoder, updates decoder on current window
5. **Dynamic Thresholding**: Combines 98th percentile and Otsu method
6. **State Persistence**: Saves model weights, window, and scores to S3 every 50 images

## Dependencies

See `requirements.txt` for the complete list. Core dependencies:
- PyTorch - Deep learning framework
- NumPy - Numerical operations
- PIL (Pillow) - Image processing
- boto3 - AWS S3 client
- torchvision - Image transformations

## License

MIT License 