import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from PIL.Image import Image as PILImage
from torchvision import transforms
from collections import deque
from io import BytesIO

from .anomaly_detector import AnomalyDetector
from bird_detector.autoencoder.convolution import ConvAutoencoder
from bird_detector.autoencoder.scores import localized_reconstruction_score
from bird_detector.autoencoder.otsu_method import otsu_method
from bird_detector.autoencoder import get_loss_function, get_optimizer


class ConvDetector(AnomalyDetector):

  def __init__(self, config, s3):
    super().__init__(config, s3) # saves config to self.config and s3 to self.s3

    # -- File names and S3 directory config --
    self.bucket = config.get('bucket_name', 'axiondm-photos')
    self.state_folder = config.get('state_folder', 'py')
    self.incubation_period = config.get('incubation_period', 200)
    self.incubation_steps = config.get('incubation_steps', 5)
    self.incubation_lr = config.get('incubation_lr', 1e-4)
    self.steps_per_image = config.get('steps_per_image', 2)
    self.lr_per_image = config.get('lr_per_image', 1e-6)

    # Lambda optimizations
    self.max_window_size = config.get('max_window_size', 200)  # Limit memory usage
    self.min_batch_size = config.get('min_batch_size', 32)  # Smaller batches for Lambda
    self.enable_training = config.get('enable_training', True)  # Option to disable training
    

    # Assume state_folder contains:
    # - weights.pth: ConvAE weights
    # - scores.npy: 
    self.scores_key = config.get('scores_key', 'scores.npy')
    self.weights_key = config.get('weights_key', 'model_weights.pth')
    self.img_window_key = config.get('img_window_key', 'img_window.npy')


    # -- Configure the Autoencoder --
    default_model_config = {
      "image_size": 64,
      "latent_dim": 2,
      "enc_channels": [16,32,64,8],
      "enc_kernel_sizes": [3, 3, 3, 3],
      "dec_channels": [64,32,16,3],
      "dec_kernel_sizes": [3, 3, 3, 3],
      "pool_kernel": 2,
      "upsample_mode": "nearest",
    }
    model_config = config.get('model', default_model_config)
    self.image_size = model_config['image_size']
    self.model = ConvAutoencoder(**model_config)
    
    # Load pretrained weights into the model
    model_weights = self._load_pytorch_weights(self.bucket, self.weights_key)
    if model_weights is not None:
        self.model.load_state_dict(model_weights)
    
    # Set device
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    
    # Initialize training components
    self.loss_fn = get_loss_function(config.get('loss_fn', 'log_mse'))
    self.optimizer = get_optimizer('adam', self.model.parameters(), lr=self.incubation_lr)

  def _load_pytorch_weights(self, bucket, key):
    """Load PyTorch model weights from S3."""
    try:
        response = self.s3.get_object(Bucket=bucket, Key=key)
        weights_data = response['Body'].read()
        return torch.load(BytesIO(weights_data), map_location=self.device)
    except self.s3.exceptions.NoSuchKey:
        print(f"Warning: No weights found at {bucket}/{key}. Starting with random weights.")
        return None

  def predict(self, img: PILImage) -> bool:
    """Predict if image contains an anomaly using rolling window adaptation."""
    
    try:
        # Transform image to tensor
        img_tensor = self._transform_image(img)
        
        # Load current state from S3
        img_window, anomaly_scores, total_images_processed = self._load_state()
        
        # Increment image count
        total_images_processed += 1
        
        # Burn-in period: always return False for first incubation_period images
        if total_images_processed <= self.incubation_period:
            # Still add image to window and calculate score for training purposes
            img_window.append(img_tensor.detach().cpu())
            anomaly_score = self._detect_anomaly(img_tensor)
            anomaly_scores.append(anomaly_score)
            
            # Train model on current window if enabled and we have enough images
            if (self.enable_training and 
                len(img_window) >= self.min_batch_size):
                self._train_on_window(img_window)
            
            # Save updated state to S3
            self._save_state(img_window, anomaly_scores, total_images_processed)
            
            # Always return False during burn-in period
            return False
        
        # Normal prediction after burn-in period
        # Add new image to rolling window (respect max window size)
        img_window.append(img_tensor.detach().cpu())
        
        # Perform anomaly detection on current image
        anomaly_score = self._detect_anomaly(img_tensor)
        anomaly_scores.append(anomaly_score)
        
        # Train model on current window if enabled and we have enough images
        if (self.enable_training and 
            len(img_window) >= self.min_batch_size):
            self._train_on_window(img_window)
        
        # Calculate dynamic threshold
        threshold = self._calculate_dynamic_threshold(anomaly_scores)
        
        # Save updated state to S3
        self._save_state(img_window, anomaly_scores, total_images_processed)
        
        # Return anomaly prediction
        return anomaly_score > threshold
        
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        # Return False (no anomaly) as fallback
        return False

  def _transform_image(self, img: PILImage) -> torch.Tensor:
    """Transform PIL image to tensor for model input."""
    # Handle both square and rectangular image sizes
    if isinstance(self.image_size, int):
        # Square image - resize to exact dimensions
        resize_size = (self.image_size, self.image_size)
    else:
        # Rectangular image - use the tuple directly
        resize_size = self.image_size
    
    transform = transforms.Compose([
        transforms.Resize(resize_size),  # Resize to model input size (height, width)
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor.to(self.device)

  def _load_state(self):
    """Load current image window and anomaly scores from S3."""
    try:
        # Load image window
        img_window = self._load_npy_from_s3(self.bucket, self.img_window_key)
        if img_window is None:
            # Initialize empty window
            img_window = deque(maxlen=self.max_window_size)
        else:
            # Convert numpy array back to deque, limit to max_window_size
            img_window = deque(img_window, maxlen=self.max_window_size)
        
        # Load anomaly scores and image count
        scores_data = self._load_npy_from_s3(self.bucket, self.scores_key)
        if scores_data is None:
            anomaly_scores = []
            total_images_processed = 0
        else:
            # Check if scores_data is a structured array with metadata
            if scores_data.dtype.names is not None and 'total_images' in scores_data.dtype.names:
                # New format with metadata
                anomaly_scores = scores_data['scores'].tolist()
                total_images_processed = int(scores_data['total_images'][0])
            else:
                # Old format - just scores array
                anomaly_scores = scores_data.tolist()
                total_images_processed = len(anomaly_scores)
        
        return img_window, anomaly_scores, total_images_processed
        
    except Exception as e:
        print(f"Error loading state from S3: {e}")
        # Return empty state as fallback
        return deque(maxlen=self.max_window_size), [], 0

  def _save_state(self, img_window, anomaly_scores, total_images_processed):
    """Save current state to S3."""
    try:
        # Save image window as numpy array
        if len(img_window) > 0:
            window_array = torch.stack(list(img_window)).cpu().numpy()
            self._save_npy_to_s3(window_array, self.bucket, self.img_window_key)
        
        # Save anomaly scores with metadata
        if len(anomaly_scores) > 0:
            # Create structured array with scores and metadata
            scores_array = np.array(anomaly_scores)
            metadata_array = np.array([(score, total_images_processed) for score in anomaly_scores], 
                                    dtype=[('scores', 'f8'), ('total_images', 'i8')])
            self._save_npy_to_s3(metadata_array, self.bucket, self.scores_key)
            
        # Save updated model weights periodically (every 50 images)
        if total_images_processed % 50 == 0:
            self._save_model_weights()
            
    except Exception as e:
        print(f"Error saving state to S3: {e}")

  def _save_model_weights(self):
    """Save current model weights to S3."""
    try:
        # Save model weights to buffer
        buffer = BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        
        # Upload to S3
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.weights_key,
            Body=buffer.getvalue()
        )
        
    except Exception as e:
        print(f"Error saving model weights to S3: {e}")

  def _detect_anomaly(self, img_tensor: torch.Tensor) -> float:
    """Detect anomaly in image using localized reconstruction scoring."""
    self.model.eval()
    
    with torch.no_grad():
        # Get model reconstruction
        output = self.model(img_tensor)
        
        # Calculate reconstruction difference
        diff = output - img_tensor
        
        # Calculate localized anomaly score
        anomaly_score = localized_reconstruction_score(
            diff_batch=diff**2,
            original_image_size=self.image_size[0],  # Use height as the size parameter
            gaussian_kernel_size=10,
            gaussian_sigma=1.0,
            device=self.device
        )
        
        # Apply log transformation
        anomaly_score = torch.log(anomaly_score)
        
        return anomaly_score.item()

  def _train_on_window(self, img_window):
    """Train model on current rolling window."""
    if len(img_window) < self.min_batch_size:
        return
    
    try:
        # Convert window to tensor dataset
        window_tensors = torch.stack(list(img_window)).detach().requires_grad_(True)
        window_dataset = torch.utils.data.TensorDataset(window_tensors, window_tensors)
        window_dataloader = torch.utils.data.DataLoader(
            window_dataset, 
            batch_size=min(self.min_batch_size, len(img_window)), 
            shuffle=True
        )
        
        # Freeze encoder for stability (only update decoder)
        self._freeze_encoder()
        
        # Train for specified number of steps
        self.model.train()
        dataloader_iter = iter(window_dataloader)
        
        for step in range(self.steps_per_image):
            try:
                batch_data, _ = next(dataloader_iter)
            except StopIteration:
                # Restart iterator if we run out of batches
                dataloader_iter = iter(window_dataloader)
                batch_data, _ = next(dataloader_iter)
            
            batch_data = batch_data.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(batch_data)
            loss = self.loss_fn(output, batch_data)
            loss.backward()
            self.optimizer.step()
        
        # Unfreeze all parameters
        self._unfreeze_all()
        self.model.eval()
        
    except Exception as e:
        print(f"Error during training: {e}")
        # Ensure model is in eval mode even if training fails
        self._unfreeze_all()
        self.model.eval()

  def _freeze_encoder(self):
    """Freeze encoder parameters for stability."""
    for param in self.model.encoder_conv.parameters():
        param.requires_grad = False
    for param in self.model.fc_enc.parameters():
        param.requires_grad = False

  def _unfreeze_all(self):
    """Unfreeze all model parameters."""
    for param in self.model.parameters():
        param.requires_grad = True

  def _calculate_dynamic_threshold(self, anomaly_scores):
    """Calculate dynamic threshold using 98th percentile or Otsu method."""
    if len(anomaly_scores) < 10:
        return 0.0  # Default threshold if not enough data
    
    scores_array = np.array(anomaly_scores)
    percentile_98 = np.percentile(scores_array, 98)
    otsu_threshold = otsu_method(scores_array)
    
    # Use the larger of the two thresholds
    dynamic_threshold = max(percentile_98, otsu_threshold)
    return dynamic_threshold

  def get_current_state(self):
    """Get current state information for monitoring."""
    try:
        img_window, anomaly_scores, total_images_processed = self._load_state()
        threshold = self._calculate_dynamic_threshold(anomaly_scores)
        
        return {
            'window_size': len(img_window),
            'total_images_processed': total_images_processed,
            'incubation_period': self.incubation_period,
            'in_burn_in_period': total_images_processed <= self.incubation_period,
            'total_scores': len(anomaly_scores),
            'current_threshold': threshold,
            'recent_scores': anomaly_scores[-10:] if len(anomaly_scores) >= 10 else anomaly_scores,
            'model_device': str(self.device),
            'training_enabled': self.enable_training
        }
    except Exception as e:
        return {'error': str(e)}

  def get_anomaly_score(self, img: PILImage) -> float:
    """Get anomaly score for an image without updating state."""
    try:
        img_tensor = self._transform_image(img)
        return self._detect_anomaly(img_tensor)
    except Exception as e:
        print(f"Error calculating anomaly score: {e}")
        return 0.0