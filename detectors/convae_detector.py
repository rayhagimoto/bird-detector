import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from PIL.Image import Image as PILImage
from torchvision import transforms
from collections import deque
from io import BytesIO
import os
import pickle

from .anomaly_detector import AnomalyDetector
from ..autoencoder.convolution import ConvAutoencoder
from ..autoencoder.scores import localized_reconstruction_score
from ..autoencoder.otsu_method import otsu_method
from ..autoencoder import get_loss_function, get_optimizer


class ConvDetector(AnomalyDetector):

  def __init__(self, config, s3):
    # Set device first!
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    super().__init__(config, s3) # saves config to self.config and s3 to self.s3
    print(f"ConvDetector initialized with config: {config}")

    # -- File names and S3 directory config --
    self.bucket = config.get('bucket_name', 'your-bucket-here')
    self.incubation_period = int(config.get('incubation_period', 200))
    self.incubation_steps = int(config.get('incubation_steps', 5))
    self.incubation_lr = float(config.get('incubation_lr', 1e-4))
    self.steps_per_image = int(config.get('steps_per_image', 2))
    self.lr_per_image = float(config.get('lr_per_image', 1e-6))

    # Lambda optimizations
    self.max_window_size = config.get('max_window_size', 200)  # Limit memory usage
    self.min_batch_size = config.get('min_batch_size', 32)  # Smaller batches for Lambda
    self.enable_training = config.get('enable_training', True)  # Option to disable training
    

    # Assume state_folder contains:
    # - weights.pth: ConvAE weights
    # - scores.npy: 
    self.weights_key = config.get('weights_key', 'py/model_weights.pth')
    self.state_dict_s3_key = config.get('state_dict_key', 'py/state_dict.pkl')


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
    self.model = ConvAutoencoder(**model_config)
    self.image_size = self.model.image_size
    
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
    print(f"Loading PyTorch weights from S3: bucket={bucket}, key={key}")
    """Load PyTorch model weights from S3."""
    try:
        response = self.s3.get_object(Bucket=bucket, Key=key)
        weights_data = response['Body'].read()
        return torch.load(BytesIO(weights_data), map_location=self.device)
    except self.s3.exceptions.NoSuchKey:
        print(f"Warning: No weights found at {bucket}/{key}. Starting with random weights.")
        return None

  def predict(self, img: PILImage) -> bool:
    print("ConvDetector.predict called")
    """Predict if image contains an anomaly using rolling window adaptation."""
    
    try:
        # Transform image to tensor
        print(f"Transforming image to tensor, original size: {img.size}")
        img_tensor = self._transform_image(img)
        print(f"Image transformed to tensor with shape: {img_tensor.shape}")
        
        # Load current state from S3
        print("Loading state from S3...")
        img_window, _, total_images_processed, state_dict = self._load_state()
        print(f"Loaded state: img_window len={len(img_window)}, total_images_processed={total_images_processed}")
        
        # Increment image count
        total_images_processed += 1

        # Add new image to rolling window (respect max window size)
        img_window.append(img_tensor.squeeze(0).detach().cpu().numpy())
        state_dict['img_window'] = list(img_window)
        state_dict['total_images_processed'] = total_images_processed
        
        # Burn-in period: always return False for first incubation_period images
        if total_images_processed <= self.incubation_period:
            print(f"Burn-in period: {total_images_processed}/{self.incubation_period}")
            # Still add image to window and calculate score for training purposes
            anomaly_scores = state_dict.get('incubation_anomaly_scores', [])
            anomaly_score = self._detect_anomaly(img_tensor)
            anomaly_scores.append(anomaly_score)
            state_dict['incubation_anomaly_scores'] = anomaly_scores
            
            # Train model on current window if enabled and we have enough images
            if (self.enable_training and 
                len(img_window) >= self.min_batch_size):
                print("Training model on current window (if enabled and enough images)...")
                self._train_on_window(img_window)
            
            # Save updated state to S3
            print("Saving state during burn-in period...")
            self._save_state(state_dict)
            print("State saved during burn-in period.")
            
            # Always return False during burn-in period
            return False
        
        # Normal prediction after burn-in period
        print("Normal prediction after burn-in period")
        
        # Perform anomaly detection on current image
        anomaly_scores = state_dict.get('anomaly_scores', [])
        anomaly_score = self._detect_anomaly(img_tensor)
        anomaly_scores.append(anomaly_score)
        state_dict['anomaly_scores'] = anomaly_scores
        print(f"Anomaly score: {anomaly_score}")
        
        # Train model on current window if enabled and we have enough images
        if (self.enable_training and 
            len(img_window) >= self.min_batch_size):
            print("Training model on current window (if enabled and enough images)...")
            self._train_on_window(img_window)
        
        # Only calculate threshold and do inference after incubation
        print(f"Calculating dynamic threshold...")
        threshold = self._calculate_dynamic_threshold(anomaly_scores)
        print(f"Dynamic threshold: {threshold}")
        
        # Save updated state to S3
        print("Saving updated state after prediction...")
        self._save_state(state_dict)
        print("State saved after prediction.")
        
        # Return anomaly prediction
        return anomaly_score > threshold
        
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        # Return False (no anomaly) as fallback
        return False

  def _transform_image(self, img: PILImage) -> torch.Tensor:
    print(f"_transform_image called, input size: {img.size}")
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
    print(f"Resized image to: {resize_size}")
    print(f"Image tensor shape after transform: {img_tensor.shape}")
    return img_tensor.to(self.device)

  def _load_state(self):
    print(f"_load_state called: bucket={self.bucket}, state_dict_key=state_dict.pkl")
    """Load current state (img_window, anomaly_scores, total_images_processed) from S3 using pickle."""
    try:
        # Load pickled state dict from S3
        state_bytes = self._load_bytes_from_s3(self.bucket, self.state_dict_s3_key)
        if state_bytes is not None:
            state_dict = pickle.loads(state_bytes)
            img_window = deque(state_dict.get('img_window', []), maxlen=self.max_window_size)
            anomaly_scores = state_dict.get('anomaly_scores', [])
            total_images_processed = state_dict.get('total_images_processed', 0)
            print(f"[DEBUG][LOAD] total_images_processed loaded from S3: {total_images_processed}")
        else:
            state_dict = {}
            img_window = deque(maxlen=self.max_window_size)
            anomaly_scores = []
            total_images_processed = 0
            print(f"[DEBUG][LOAD] No state_dict found in S3. total_images_processed set to 0.")
        print(f"Loaded img_window (len={len(img_window)}), anomaly_scores (len={len(anomaly_scores)}), total_images_processed={total_images_processed}")
        return img_window, anomaly_scores, total_images_processed, state_dict
    except Exception as e:
        print(f"Error loading state from S3: {e}")
        print(f"[DEBUG][LOAD] Exception occurred. total_images_processed set to 0.")
        return deque(maxlen=self.max_window_size), [], 0, {}

  def _save_state(self, state_dict):
    print(f"_save_state called: bucket={self.bucket}, state_dict_key=state_dict.pkl")
    print(f"[DEBUG][SAVE] total_images_processed to be saved to S3: {state_dict.get('total_images_processed')}")
    print(f"img_window type: {type(state_dict.get('img_window'))}")
    print(f"img_window length: {len(state_dict.get('img_window', []))}")
    print(f"img_window contents: {[x.shape if hasattr(x, 'shape') else type(x) for x in state_dict.get('img_window', [])]}")
    print("[INFO] total_images_processed is updated in the predict() method: it is incremented by 1 each time a new image is processed.")
    """Save current state to S3 using pickle."""
    try:
        state_bytes = pickle.dumps(state_dict)
        self._save_bytes_to_s3(state_bytes, self.bucket, self.state_dict_s3_key)
        print(f"[DEBUG] state_dict saved to S3 at {self.state_dict_s3_key}")
        self._save_model_weights()
        print(f"[DEBUG] model_weights saved to S3 at {self.weights_key}")
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
        window_tensors = torch.from_numpy(np.stack(list(img_window), axis=0)).detach().requires_grad_(True)
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
        img_window, anomaly_scores, total_images_processed, state_dict = self._load_state()
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

  def _load_bytes_from_s3(self, bucket, key):
    try:
        response = self.s3.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except Exception as e:
        print(f"Error loading bytes from S3: {e}")
        return None

  def _save_bytes_to_s3(self, data_bytes, bucket, key):
    try:
        self.s3.put_object(Bucket=bucket, Key=key, Body=data_bytes)
    except Exception as e:
        print(f"Error saving bytes to S3: {e}")