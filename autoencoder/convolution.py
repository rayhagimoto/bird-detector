import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Callable, Union

class ConvAutoencoder(nn.Module):
    """
    Modular Convolutional Autoencoder for anomaly detection in images.
    Allows flexible configuration of encoder/decoder layer depths, kernel sizes, and output channels.

    Args:
        image_size (Union[int, Tuple[int, int], List[int]]): Input image size. 
            If int: square image (e.g., 64 for 64x64)
            If tuple/list: rectangular image (e.g., (64, 48) for 64x48)
        latent_dim (int): Dimension of latent space
        enc_channels (List[int]): Output channels for each encoder conv layer (e.g., [16, 32, 64, 8])
        enc_kernel_sizes (List[int]): Kernel sizes for each encoder conv layer (e.g., [3, 3, 3, 3])
        dec_channels (List[int]): Output channels for each decoder conv layer (reverse of encoder, e.g., [64, 32, 16, 3])
        dec_kernel_sizes (List[int]): Kernel sizes for each decoder conv layer (e.g., [3, 3, 3, 3])
        pool_kernel (int): Kernel size for MaxPool2d (default: 2)
        upsample_mode (str): Mode for nn.Upsample (default: 'nearest')
    """
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int], List[int]],
        latent_dim: int,
        enc_channels: List[int] = [16, 32, 64, 8],
        enc_kernel_sizes: List[int] = [3, 3, 3, 3],
        dec_channels: List[int] = [64, 32, 16, 3],
        dec_kernel_sizes: List[int] = [3, 3, 3, 3],
        pool_kernel: int = 2,
        upsample_mode: str = 'nearest',
    ):
        super().__init__()
        
        # Handle image_size input
        if isinstance(image_size, int):
            self.image_height = image_size
            self.image_width = image_size
        elif isinstance(image_size, list):
            if len(image_size) != 2:
                raise ValueError("image_size list must have exactly 2 elements")
            self.image_height, self.image_width = tuple(image_size)
        elif isinstance(image_size, tuple):
            if len(image_size) != 2:
                raise ValueError("image_size tuple must have exactly 2 elements")
            self.image_height, self.image_width = image_size
        else:
            raise ValueError("image_size must be int, tuple, or list")
        
        # Store original image_size for backward compatibility
        self.image_size = (self.image_height, self.image_width)
        
        # Check divisibility for both dimensions
        total_pooling_factor = pool_kernel ** len(enc_channels)
        assert self.image_height % total_pooling_factor == 0, f"image_height ({self.image_height}) must be divisible by total pooling factor ({total_pooling_factor})"
        assert self.image_width % total_pooling_factor == 0, f"image_width ({self.image_width}) must be divisible by total pooling factor ({total_pooling_factor})"
        
        self.latent_dim = latent_dim
        self.enc_channels = enc_channels
        self.enc_kernel_sizes = enc_kernel_sizes
        self.dec_channels = dec_channels
        self.dec_kernel_sizes = dec_kernel_sizes
        self.pool_kernel = pool_kernel
        self.upsample_mode = upsample_mode

        # Encoder
        enc_layers = []
        in_ch = 3
        curr_height = self.image_height
        curr_width = self.image_width
        for out_ch, k in zip(enc_channels, enc_kernel_sizes):
            enc_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k//2))
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.MaxPool2d(pool_kernel))
            in_ch = out_ch
            curr_height //= pool_kernel
            curr_width //= pool_kernel
        self.encoder_conv = nn.Sequential(*enc_layers)
        self.flat_size = curr_height * curr_width * enc_channels[-1]
        self.fc_enc = nn.Linear(self.flat_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_size)

        # Decoder
        dec_layers = []
        in_ch = enc_channels[-1]
        curr_height = curr_height
        curr_width = curr_width
        for i, (out_ch, k) in enumerate(zip(dec_channels, dec_kernel_sizes)):
            dec_layers.append(nn.Upsample(scale_factor=pool_kernel, mode=upsample_mode))
            dec_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k//2))
            if i < len(dec_channels) - 1:
                dec_layers.append(nn.ReLU())
            else:
                dec_layers.append(nn.Sigmoid())
            in_ch = out_ch
            curr_height *= pool_kernel
            curr_width *= pool_kernel
        self.decoder_conv = nn.Sequential(*dec_layers)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        z = self.fc_enc(x)
        return z

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_dec(z)
        # Calculate the height and width after encoding
        encoded_height = self.image_height // (self.pool_kernel ** len(self.enc_channels))
        encoded_width = self.image_width // (self.pool_kernel ** len(self.enc_channels))
        x = x.view(x.size(0), self.enc_channels[-1], encoded_height, encoded_width)
        x = self.decoder_conv(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) 