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
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass

from .anomaly_detector import AnomalyDetector
from ..autoencoder.convolution import ConvAutoencoder
from ..autoencoder.scores import localized_reconstruction_score
from ..autoencoder.otsu_method import otsu_method
from ..autoencoder import get_loss_function, get_optimizer

@dataclass
class ConvAEConfig:
    """Configuration for ConvAE anomaly detection."""
    # Model configuration
    image_size: Union[int, Tuple[int, int]] = 64
    latent_dim: int = 2
    enc_channels: List[int] = None
    enc_kernel_sizes: List[int] = None
    dec_channels: List[int] = None
    dec_kernel_sizes: List[int] = None
    pool_kernel: int = 2
    upsample_mode: str = "nearest"
    
    # Training configuration
    incubation_period: int = 200
    incubation_steps: int = 5
    incubation_lr: float = 1e-4
    steps_per_image: int = 2
    lr_per_image: float = 1e-6
    loss_fn: str = "log_mse"
    optimizer_type: str = "adam"
    
    # Window and batch configuration
    max_window_size: int = 200
    min_batch_size: int = 32
    enable_training: bool = True
    
    # Anomaly detection configuration
    gaussian_kernel_size: int = 10
    gaussian_sigma: float = 1.0
    percentile_threshold: float = 98.0
    min_scores_for_threshold: int = 10
    
    def __post_init__(self):
        """Set default values for list fields."""
        if self.enc_channels is None:
            self.enc_channels = [16, 32, 64, 8]
        if self.enc_kernel_sizes is None:
            self.enc_kernel_sizes = [3, 3, 3, 3]
        if self.dec_channels is None:
            self.dec_channels = [64, 32, 16, 3]
        if self.dec_kernel_sizes is None:
            self.dec_kernel_sizes = [3, 3, 3, 3]

@dataclass
class ConvAEState:
    """State for ConvAE anomaly detection."""
    img_window: deque
    anomaly_scores: List[float]
    incubation_anomaly_scores: List[float]
    total_images_processed: int
    model_state_dict: Optional[Dict[str, Any]] = None

@dataclass
class ConvAEResult:
    """Result of ConvAE anomaly detection."""
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    is_burn_in: bool
    total_processed: int
    reason: str

def create_model(config: ConvAEConfig, device: torch.device) -> ConvAutoencoder:
    """Create and initialize ConvAutoencoder model."""
    model_config = {
        "image_size": config.image_size,
        "latent_dim": config.latent_dim,
        "enc_channels": config.enc_channels,
        "enc_kernel_sizes": config.enc_kernel_sizes,
        "dec_channels": config.dec_channels,
        "dec_kernel_sizes": config.dec_kernel_sizes,
        "pool_kernel": config.pool_kernel,
        "upsample_mode": config.upsample_mode,
    }
    model = ConvAutoencoder(**model_config)
    model.to(device)
    return model

def transform_image(image: PILImage, image_size: Union[int, Tuple[int, int]], device: torch.device) -> torch.Tensor:
    """Transform PIL image to tensor for model input."""
    # Handle both square and rectangular image sizes
    if isinstance(image_size, int):
        resize_size = (image_size, image_size)
    else:
        resize_size = image_size
    
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor.to(device)

def calculate_anomaly_score(
    model: ConvAutoencoder,
    img_tensor: torch.Tensor,
    config: ConvAEConfig,
    device: torch.device
) -> float:
    """Calculate anomaly score for an image using the model."""
    model.eval()
    
    with torch.no_grad():
        # Get model reconstruction
        output = model(img_tensor)
        
        # Calculate reconstruction difference
        diff = output - img_tensor
        
        # Get image size for scoring
        if isinstance(config.image_size, int):
            image_size_param = config.image_size
        else:
            image_size_param = config.image_size[0]  # Use height
        
        # Calculate localized anomaly score
        anomaly_score = localized_reconstruction_score(
            diff_batch=diff**2,
            original_image_size=image_size_param,
            gaussian_kernel_size=config.gaussian_kernel_size,
            gaussian_sigma=config.gaussian_sigma,
            device=device
        )
        
        # Apply log transformation
        anomaly_score = torch.log(anomaly_score)
        
        return anomaly_score.item()

def calculate_dynamic_threshold(
    anomaly_scores: List[float],
    config: ConvAEConfig
) -> float:
    """Calculate dynamic threshold using percentile and Otsu method."""
    if len(anomaly_scores) < config.min_scores_for_threshold:
        return 0.0
    
    scores_array = np.array(anomaly_scores)
    percentile_threshold = np.percentile(scores_array, config.percentile_threshold)
    otsu_threshold = otsu_method(scores_array)
    
    # Use the larger of the two thresholds
    return max(percentile_threshold, otsu_threshold)

def freeze_encoder(model: ConvAutoencoder) -> None:
    """Freeze encoder parameters for stability."""
    for param in model.encoder_conv.parameters():
        param.requires_grad = False
    for param in model.fc_enc.parameters():
        param.requires_grad = False

def unfreeze_all(model: ConvAutoencoder) -> None:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True

def train_on_window(
    model: ConvAutoencoder,
    img_window: deque,
    config: ConvAEConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module
) -> None:
    """Train model on current rolling window."""
    if len(img_window) < config.min_batch_size:
        return
    
    try:
        # Convert window to tensor dataset
        window_tensors = torch.from_numpy(np.stack(list(img_window), axis=0)).detach().requires_grad_(True)
        window_dataset = torch.utils.data.TensorDataset(window_tensors, window_tensors)
        window_dataloader = torch.utils.data.DataLoader(
            window_dataset, 
            batch_size=min(config.min_batch_size, len(img_window)), 
            shuffle=True
        )
        
        # Freeze encoder for stability
        freeze_encoder(model)
        
        # Train for specified number of steps
        model.train()
        dataloader_iter = iter(window_dataloader)
        
        for step in range(config.steps_per_image):
            try:
                batch_data, _ = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(window_dataloader)
                batch_data, _ = next(dataloader_iter)
            
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            output = model(batch_data)
            loss = loss_fn(output, batch_data)
            loss.backward()
            optimizer.step()
        
        # Unfreeze all parameters
        unfreeze_all(model)
        model.eval()
        
    except Exception as e:
        print(f"Error during training: {e}")
        unfreeze_all(model)
        model.eval()

def process_image(
    image: PILImage,
    model: ConvAutoencoder,
    state: Optional[ConvAEState],
    config: ConvAEConfig,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_fn: Optional[nn.Module] = None
) -> Tuple[ConvAEResult, ConvAEState]:
    """Process a single image for anomaly detection using functional approach."""
    
    # Transform image to tensor
    img_tensor = transform_image(image, config.image_size, device)
    
    # Initialize or update state
    if state is None:
        img_window = deque(maxlen=config.max_window_size)
        anomaly_scores = []
        incubation_anomaly_scores = []
        total_images_processed = 0
    else:
        img_window = state.img_window
        anomaly_scores = state.anomaly_scores.copy()
        incubation_anomaly_scores = state.incubation_anomaly_scores.copy()
        total_images_processed = state.total_images_processed
    
    # Increment image count
    total_images_processed += 1
    
    # Add new image to rolling window
    img_window.append(img_tensor.squeeze(0).detach().cpu().numpy())
    
    # Calculate anomaly score
    anomaly_score = calculate_anomaly_score(model, img_tensor, config, device)
    
    # Determine if we're in burn-in period
    is_burn_in = total_images_processed <= config.incubation_period
    
    if is_burn_in:
        # During burn-in period
        incubation_anomaly_scores.append(anomaly_score)
        
        # Train model if enabled and we have enough images
        if (config.enable_training and 
            len(img_window) >= config.min_batch_size and 
            optimizer is not None and 
            loss_fn is not None):
            train_on_window(model, img_window, config, device, optimizer, loss_fn)
        
        # Create result
        result = ConvAEResult(
            is_anomaly=False,
            anomaly_score=anomaly_score,
            threshold=0.0,
            is_burn_in=True,
            total_processed=total_images_processed,
            reason=f"Burn-in period: {total_images_processed}/{config.incubation_period}"
        )
        
    else:
        # Normal prediction after burn-in period
        anomaly_scores.append(anomaly_score)
        
        # Train model if enabled and we have enough images
        if (config.enable_training and 
            len(img_window) >= config.min_batch_size and 
            optimizer is not None and 
            loss_fn is not None):
            train_on_window(model, img_window, config, device, optimizer, loss_fn)
        
        # Calculate threshold
        threshold = calculate_dynamic_threshold(anomaly_scores, config)
        
        # Determine if anomaly
        is_anomaly = anomaly_score > threshold
        
        # Create result
        result = ConvAEResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            threshold=threshold,
            is_burn_in=False,
            total_processed=total_images_processed,
            reason=f"Anomaly detected: {is_anomaly} (score: {anomaly_score:.4f}, threshold: {threshold:.4f})"
        )
    
    # Create updated state
    updated_state = ConvAEState(
        img_window=img_window,
        anomaly_scores=anomaly_scores,
        incubation_anomaly_scores=incubation_anomaly_scores,
        total_images_processed=total_images_processed,
        model_state_dict=model.state_dict() if model else None
    )
    
    return result, updated_state

def process_image_batch(
    images: List[PILImage],
    model: ConvAutoencoder,
    config: ConvAEConfig,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_fn: Optional[nn.Module] = None
) -> Tuple[List[ConvAEResult], ConvAEState]:
    """Process a batch of images for anomaly detection."""
    state = None
    results = []
    
    for image in images:
        result, state = process_image(image, model, state, config, device, optimizer, loss_fn)
        results.append(result)
    
    return results, state

def get_anomaly_score_only(
    image: PILImage,
    model: ConvAutoencoder,
    config: ConvAEConfig,
    device: torch.device
) -> float:
    """Get anomaly score for an image without updating state."""
    img_tensor = transform_image(image, config.image_size, device)
    return calculate_anomaly_score(model, img_tensor, config, device)

def create_optimizer_and_loss(
    model: ConvAutoencoder,
    config: ConvAEConfig
) -> Tuple[torch.optim.Optimizer, nn.Module]:
    """Create optimizer and loss function for training."""
    optimizer = get_optimizer(config.optimizer_type, model.parameters(), lr=config.incubation_lr)
    loss_fn = get_loss_function(config.loss_fn)
    return optimizer, loss_fn

class ConvDetector(AnomalyDetector):
    """S3-backed ConvAE anomaly detector (original implementation)."""

    def __init__(self, config, s3):
        # Set device first!
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(config, s3)
        print(f"ConvDetector initialized with config: {config}")

        # S3 configuration
        self.bucket = config.get('bucket_name', 'your-bucket-here')
        self.weights_key = config.get('weights_key', 'py/model_weights.pth')
        self.state_dict_s3_key = config.get('state_dict_key', 'py/state_dict.pkl')

        # Create ConvAEConfig from dict config
        model_config = config.get('model', {})
        self.convae_config = ConvAEConfig(
            image_size=model_config.get('image_size', 64),
            latent_dim=model_config.get('latent_dim', 2),
            enc_channels=model_config.get('enc_channels', [16, 32, 64, 8]),
            enc_kernel_sizes=model_config.get('enc_kernel_sizes', [3, 3, 3, 3]),
            dec_channels=model_config.get('dec_channels', [64, 32, 16, 3]),
            dec_kernel_sizes=model_config.get('dec_kernel_sizes', [3, 3, 3, 3]),
            pool_kernel=model_config.get('pool_kernel', 2),
            upsample_mode=model_config.get('upsample_mode', "nearest"),
            incubation_period=int(config.get('incubation_period', 200)),
            incubation_steps=int(config.get('incubation_steps', 5)),
            incubation_lr=float(config.get('incubation_lr', 1e-4)),
            steps_per_image=int(config.get('steps_per_image', 2)),
            lr_per_image=float(config.get('lr_per_image', 1e-6)),
            loss_fn=config.get('loss_fn', 'log_mse'),
            optimizer_type=config.get('optimizer_type', 'adam'),
            max_window_size=config.get('max_window_size', 200),
            min_batch_size=config.get('min_batch_size', 32),
            enable_training=config.get('enable_training', True)
        )

        # Create model
        self.model = create_model(self.convae_config, self.device)
        self.image_size = self.model.image_size
        
        # Load pretrained weights
        model_weights = self._load_pytorch_weights(self.bucket, self.weights_key)
        if model_weights is not None:
            self.model.load_state_dict(model_weights)
        
        # Initialize training components
        self.optimizer, self.loss_fn = create_optimizer_and_loss(self.model, self.convae_config)

    def _load_pytorch_weights(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        """Load PyTorch model weights from S3."""
        print(f"Loading PyTorch weights from S3: bucket={bucket}, key={key}")
        try:
            response = self.s3.get_object(Bucket=bucket, Key=key)
            weights_data = response['Body'].read()
            return torch.load(BytesIO(weights_data), map_location=self.device)
        except self.s3.exceptions.NoSuchKey:
            print(f"Warning: No weights found at {bucket}/{key}. Starting with random weights.")
            return None

    def _load_state_dict(self) -> Optional[Dict[str, Any]]:
        """Load state dictionary from S3."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=self.state_dict_s3_key)
            state_dict = pickle.load(BytesIO(response['Body'].read()))
            return state_dict
        except Exception:
            return None

    def _save_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Save state dictionary to S3."""
        try:
            buf = BytesIO()
            pickle.dump(state_dict, buf)
            buf.seek(0)
            self.s3.put_object(Bucket=self.bucket, Key=self.state_dict_s3_key, Body=buf.getvalue())
        except Exception as e:
            print(f"Error saving state to S3: {e}")

    def _save_model_weights(self) -> None:
        """Save current model weights to S3."""
        try:
            buffer = BytesIO()
            torch.save(self.model.state_dict(), buffer)
            buffer.seek(0)
            
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self.weights_key,
                Body=buffer.getvalue()
            )
        except Exception as e:
            print(f"Error saving model weights to S3: {e}")

    def _load_scores_from_s3(self, scores_type: str = "anomaly") -> Optional[List[float]]:
        """Load scores from S3 using base class helper."""
        scores_key = f"{self.state_folder}/{scores_type}_scores.csv"
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=scores_key)
            lines = response['Body'].read().decode('utf-8').splitlines()
            return [float(line.strip()) for line in lines if line.strip()]
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception:
            return None

    def _save_scores_to_s3(self, scores: List[float], scores_type: str = "anomaly") -> None:
        """Save scores to S3 using base class helper."""
        scores_key = f"{self.state_folder}/{scores_type}_scores.csv"
        self._save_csv_to_s3(scores, self.bucket, scores_key)

    def _state_to_dict(self, state: ConvAEState) -> Dict[str, Any]:
        """Convert ConvAEState to dictionary for S3 storage."""
        return {
            'img_window': list(state.img_window),
            'anomaly_scores': state.anomaly_scores,
            'incubation_anomaly_scores': state.incubation_anomaly_scores,
            'total_images_processed': state.total_images_processed
        }

    def _dict_to_state(self, state_dict: Dict[str, Any]) -> ConvAEState:
        """Convert dictionary from S3 to ConvAEState."""
        return ConvAEState(
            img_window=deque(state_dict.get('img_window', []), maxlen=self.convae_config.max_window_size),
            anomaly_scores=state_dict.get('anomaly_scores', []),
            incubation_anomaly_scores=state_dict.get('incubation_anomaly_scores', []),
            total_images_processed=state_dict.get('total_images_processed', 0)
        )

    def predict(self, img: PILImage) -> bool:
        """Predict if image contains an anomaly using S3-backed state."""
        print("ConvDetector.predict called")
        
        try:
            # Load state from S3
            print("Loading state from S3...")
            state_dict = self._load_state_dict()
            state = self._dict_to_state(state_dict) if state_dict else None
            
            # Process image using functional approach
            result, updated_state = process_image(
                img, self.model, state, self.convae_config, self.device, 
                self.optimizer, self.loss_fn
            )
            
            # Save updated state to S3
            print("Saving updated state to S3...")
            self._save_state_dict(self._state_to_dict(updated_state))
            self._save_model_weights()
            
            # Print logging information
            print(f"[ConvDetector] Processing image #{updated_state.total_images_processed}")
            print(f"[ConvDetector] {result.reason}")
            
            return result.is_anomaly
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return False

    def get_anomaly_score(self, img: PILImage) -> float:
        """Get anomaly score for an image without updating state."""
        try:
            return get_anomaly_score_only(img, self.model, self.convae_config, self.device)
        except Exception as e:
            print(f"Error calculating anomaly score: {e}")
            return 0.0

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state information for monitoring."""
        try:
            state_dict = self._load_state_dict()
            if state_dict is None:
                return {'error': 'No state found'}
            
            state = self._dict_to_state(state_dict)
            threshold = calculate_dynamic_threshold(state.anomaly_scores, self.convae_config)
            
            return {
                'window_size': len(state.img_window),
                'total_images_processed': state.total_images_processed,
                'incubation_period': self.convae_config.incubation_period,
                'in_burn_in_period': state.total_images_processed <= self.convae_config.incubation_period,
                'total_scores': len(state.anomaly_scores),
                'current_threshold': threshold,
                'recent_scores': state.anomaly_scores[-10:] if len(state.anomaly_scores) >= 10 else state.anomaly_scores,
                'model_device': str(self.device),
                'training_enabled': self.convae_config.enable_training
            }
        except Exception as e:
            return {'error': str(e)}

    # Legacy methods for backward compatibility
    def _load_state(self):
        """Legacy method for backward compatibility."""
        state_dict = self._load_state_dict()
        if state_dict is None:
            return deque(maxlen=self.convae_config.max_window_size), [], 0, {}
        
        state = self._dict_to_state(state_dict)
        return state.img_window, state.anomaly_scores, state.total_images_processed, state_dict

    def _save_state(self, state_dict: Dict[str, Any]) -> None:
        """Legacy method for backward compatibility."""
        self._save_state_dict(state_dict)

    def _transform_image(self, img: PILImage) -> torch.Tensor:
        """Legacy method for backward compatibility."""
        return transform_image(img, self.convae_config.image_size, self.device)

    def _detect_anomaly(self, img_tensor: torch.Tensor) -> float:
        """Legacy method for backward compatibility."""
        return calculate_anomaly_score(self.model, img_tensor, self.convae_config, self.device)

    def _train_on_window(self, img_window: deque) -> None:
        """Legacy method for backward compatibility."""
        train_on_window(self.model, img_window, self.convae_config, self.device, self.optimizer, self.loss_fn)

    def _freeze_encoder(self) -> None:
        """Legacy method for backward compatibility."""
        freeze_encoder(self.model)

    def _unfreeze_all(self) -> None:
        """Legacy method for backward compatibility."""
        unfreeze_all(self.model)

    def _calculate_dynamic_threshold(self, anomaly_scores: List[float]) -> float:
        """Legacy method for backward compatibility."""
        return calculate_dynamic_threshold(anomaly_scores, self.convae_config)