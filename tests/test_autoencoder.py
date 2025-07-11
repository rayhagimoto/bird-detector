"""
Tests for autoencoder module.
"""

import pytest
import torch
from bird_detector.autoencoder import ConvAutoencoder, SimpleAutoencoder


class TestConvAutoencoder:
    """Test ConvAutoencoder functionality."""
    
    def test_conv_autoencoder_creation(self):
        """Test that ConvAutoencoder can be created with valid parameters."""
        model = ConvAutoencoder(
            image_size=64,
            latent_dim=2,
            enc_channels=[16, 32, 64, 8],
            enc_kernel_sizes=[3, 3, 3, 3],
            dec_channels=[64, 32, 16, 3],
            dec_kernel_sizes=[3, 3, 3, 3],
        )
        assert model is not None
        assert model.latent_dim == 2
    
    def test_conv_autoencoder_forward(self):
        """Test ConvAutoencoder forward pass."""
        model = ConvAutoencoder(image_size=64, latent_dim=2)
        x = torch.randn(1, 3, 64, 64)
        output = model(x)
        assert output.shape == (1, 3, 64, 64)


class TestSimpleAutoencoder:
    """Test SimpleAutoencoder functionality."""
    
    def test_simple_autoencoder_creation(self):
        """Test that SimpleAutoencoder can be created."""
        model = SimpleAutoencoder(
            input_size=64*64*3,
            layer_sizes=[512, 256, 128],
            latent_dim=2
        )
        assert model is not None
        assert model.latent_dim == 2 