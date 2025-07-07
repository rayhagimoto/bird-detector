# Bird Detector - OpenCV Pipeline

A lightweight bird detection library optimized for AWS Lambda deployment with minimal dependencies.

## Overview

This branch contains the **OpenCV Detector**: an OpenCV-based anomaly detection pipeline that requires **no pre-training** and works out-of-the-box on new data.

## OpenCV Detector (No Pre-Training Required)

The **OpenCVDetector** uses a classic computer vision approach:
- **No pre-training or ML model is required.**
- Maintains an **Exponential Moving Average (EMA)** of previous images as a background model.
- For each new image:
  1. The image is resized and normalized.
  2. The EMA is updated with the new image.
  3. The luminance difference between the current image and EMA is computed.
  4. The difference map is blurred and thresholded to create a binary mask.
  5. Contours are extracted and filtered by area, aspect ratio, vertical position, and contrast.
  6. If a valid contour is found, the image is flagged as an anomaly (e.g., a bird is present).
- The EMA is saved to S3 after each image, so the background model is persistent across invocations.
- Detected anomaly image keys are appended to `anomalies.csv` in S3.

### Configuration Example

```
bucket_name: axiondm-photos
state_folder: py
ema_filename: ema.npy
image_size: [128, 128]
alpha: 0.05
threshold: 0.15
blur_sigma: 0.5
min_area_frac: 0.0015
max_area_frac: 0.05
exclude_bottom: true
aspect_ratio_max: 2.0
preferred_vertical_range: [0.15, 0.85]
min_contrast: 10
```

### Key Features
- **No ML training required**: Works immediately on new deployments.
- **Fast and lightweight**: Suitable for serverless and edge environments.
- **Configurable**: All detection parameters are set via YAML config.
- **Persistent background model**: EMA is updated and saved after each image.

## Usage

### OpenCV Detector

```python
from bird_detector.detectors.opencv_detector import OpenCVDetector
import boto3

s3_client = boto3.client('s3')
config = {...}  # See above

detector = OpenCVDetector(config, s3_client)
result = detector.predict(pil_image)
print(f"Anomaly detected: {result}")
```

## License

MIT License 
