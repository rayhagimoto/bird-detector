"""
Autoencoder architectures and experiment tracking for bird-watching anomaly detection.

This module provides autoencoder implementations for anomaly detection in bird images.
The autoencoders learn to compress and reconstruct normal images, allowing detection
of anomalous images based on high reconstruction error.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import copy
from torchvision import transforms


class SimpleAutoencoder(nn.Module):
    """Simple fully-connected Autoencoder for anomaly detection. Implemented as 
    a PyTorch NN.
    
    This autoencoder compresses images to a low-dimensional latent space and then
    reconstructs them. Images with high reconstruction error are flagged as anomalies.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                ENCODER                                     │
    │  Input Image (3×64×64) → Flatten (12288) → FC+ReLU → FC+ReLU → FC+ReLU →  │
    │  FC (no activation) → Latent Space (2D)                                    │
    │                                                                             │
    │                                DECODER                                     │
    │  Latent Space (2D) → FC+ReLU → FC+ReLU → FC+ReLU → FC+Sigmoid → Reshape →  │
    │  Output Image (3×64×64)                                                    │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    Example with layer_sizes=[128, 32, 8]:
    - Encoder: 12288 → 128 → 32 → 8 → 2 (latent)
    - Decoder: 2 (latent) → 8 → 32 → 128 → 12288 → (3×64×64)
    
    Args:
        input_size (int): Size of input images (assumed square, e.g., 64 for 64×64)
        latent_dim (int): Dimension of latent space (e.g., 2 for 2D visualization)
        layer_sizes (List[int]): List of hidden layer sizes for encoder/decoder.
                                Default: [128, 32, 8]
        metric (Union[Callable, str]): Metric function or predefined metric name.
                                      Options: 'mse', 'log_mse', or custom callable.
                                      If callable, signature must be: metric(x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor
    
    Attributes:
        input_size (int): Size of input images
        flattened_size (int): Size of flattened input (3 * input_size^2)
        latent_dim (int): Dimension of latent space
        layer_sizes (List[int]): Hidden layer sizes
        encoder (nn.Sequential): Encoder network
        decoder (nn.Sequential): Decoder network
        _compute_reconstruction_error: Method for computing reconstruction error
    
    Example:
        >>> model = SimpleAutoencoder(input_size=64, latent_dim=2, metric='log_mse')
        >>> x = torch.randn(1, 3, 64, 64)  # Batch of 1 image
        >>> reconstructed = model(x)  # Shape: (1, 3, 64, 64)
        >>> latent = model.encoder(x)  # Shape: (1, 2)
        >>> error = model.get_reconstruction_error(x)  # log10(MSE) per image
    """
    
    def __init__(self, input_size: int, latent_dim: int, layer_sizes: List[int], metric: Optional[Union[Callable, str]] = None, **kwargs):
        super().__init__()
        
        # Calculate flattened input size (e.g., 64×64×3 = 12288)
        self.input_size = input_size
        self.flattened_size = 3 * input_size * input_size
        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes
        
        # Define reconstruction error computation method based on metric
        if callable(metric):
            # User provided a custom function
            self._compute_reconstruction_error = metric
        elif metric == 'mse':
            self._compute_reconstruction_error = self._compute_mse
        elif metric == 'log_mse':
            self._compute_reconstruction_error = self._compute_log_mse
        else:
            print("[WARNING] No metric supplied, SimpleAutoencoder.get_reconstruction_error will not work.")
        
        # Build encoder and decoder networks
        self.encoder = self._encoder_from_layer_sizes()
        self.decoder = self._decoder_from_layer_sizes()

    def _encoder_from_layer_sizes(self) -> nn.Sequential:
        """Build the encoder network from layer sizes.
        
        Creates a sequence of fully connected layers with ReLU activations,
        ending with a linear layer to the latent space (no activation).
        
        Architecture:
        Flatten → FC+ReLU → FC+ReLU → ... → FC (no activation) → Latent
        
        Returns:
            nn.Sequential: Encoder network
        """
        layers = []
        
        # Add flatten layer first
        layers.append(nn.Flatten())
        
        # Build intermediate layers
        current_size = self.flattened_size
        for size in self.layer_sizes:
            layers.append(nn.Linear(current_size, size))
            layers.append(nn.ReLU())
            current_size = size
        
        # Add final layer to latent space (no activation)
        layers.append(nn.Linear(current_size, self.latent_dim))
        
        return nn.Sequential(*layers)
    
    def _decoder_from_layer_sizes(self) -> nn.Sequential:
        """Build the decoder network from layer sizes (symmetric to encoder).
        
        Creates a sequence of fully connected layers with ReLU activations,
        ending with a linear layer + sigmoid for image reconstruction.
        
        Architecture:
        Latent → FC+ReLU → FC+ReLU → ... → FC+Sigmoid → Reshape
        
        Returns:
            nn.Sequential: Decoder network
        """
        layers = []
        
        # Build intermediate layers (reverse order of encoder)
        current_size = self.latent_dim
        for size in reversed(self.layer_sizes):
            layers.append(nn.Linear(current_size, size))
            layers.append(nn.ReLU())
            current_size = size
        
        # Add final layer to output space with sigmoid activation
        layers.append(nn.Linear(current_size, self.flattened_size))
        layers.append(nn.Sigmoid())  # Output values in [0, 1] for image reconstruction
        
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, input_size, input_size)
        
        Returns:
            torch.Tensor: Reconstructed images of shape (batch_size, 3, input_size, input_size)
        """
        # Encode to latent space
        encoded = self.encoder(x)
        
        # Decode from latent space
        decoded = self.decoder(encoded)
        
        # Reshape back to image dimensions
        decoded = decoded.view(-1, 3, self.input_size, self.input_size)
        
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images to latent space.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, input_size, input_size)
        
        Returns:
            torch.Tensor: Latent representations of shape (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representations back to images.
        
        Args:
            z (torch.Tensor): Latent representations of shape (batch_size, latent_dim)
        
        Returns:
            torch.Tensor: Reconstructed images of shape (batch_size, 3, input_size, input_size)
        """
        decoded = self.decoder(z)
        return decoded.view(-1, 3, self.input_size, self.input_size)
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error for anomaly detection.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, input_size, input_size)
        
        Returns:
            torch.Tensor: Reconstruction error per image of shape (batch_size,)
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            return self._compute_reconstruction_error(x, reconstructed)
    
    def _compute_mse(self, x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute Mean Squared Error between input and reconstruction.
        
        Args:
            x (torch.Tensor): Original input images
            reconstructed (torch.Tensor): Reconstructed images
            
        Returns:
            torch.Tensor: MSE per image of shape (batch_size,)
        """
        return torch.mean((x - reconstructed) ** 2, dim=[1, 2, 3])
    
    def _compute_log_mse(self, x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute log10 of Mean Squared Error between input and reconstruction.
        
        Args:
            x (torch.Tensor): Original input images
            reconstructed (torch.Tensor): Reconstructed images
            
        Returns:
            torch.Tensor: log10(MSE) per image of shape (batch_size,)
        """
        mse = torch.mean((x - reconstructed) ** 2, dim=[1, 2, 3])
        # Add small epsilon to avoid log(0)
        return torch.log10(mse + 1e-8)