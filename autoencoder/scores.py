import torch
import torch.nn as nn
import numpy as np
import math

# Helper Function for Gaussian Kernel
def gaussian_kernel_2d(kernel_size, sigma, in_channels, device):
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
    kernel_tensor = kernel_tensor.repeat(in_channels, 1, 1, 1)
    return kernel_tensor.to(device)


def localized_reconstruction_score(
    diff_batch: torch.Tensor,
    original_image_size: int,
    gaussian_kernel_size: int = 10,
    gaussian_sigma: float = 1.0,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Calculates an anomaly score by identifying the peak intensity within a downsampled,
    smoothed reconstruction error map.

    This score is derived from:
    1.  Averaging input `diff` channels to a single grayscale error map.
    2.  Applying Gaussian smoothing to the error map.
    3.  Sequentially max-pooling the smoothed error map down to a 4x4 spatial resolution.
    4.  Flattening the 4x4 map and extracting the maximum value as the anomaly score.
    """
    N, C, H, W = diff_batch.shape

    # Convert multi-channel difference map to a single-channel grayscale error map.
    if C >= 2:
        processed_diff_for_blur = 1.0 / 3.0 * (diff_batch[:, 0] + diff_batch[:, 1] + diff_batch[:, 1])
    else:
        processed_diff_for_blur = diff_batch.squeeze(1)

    conv_input = processed_diff_for_blur.unsqueeze(1)

    # Apply Gaussian smoothing to blur the error map, making localized peaks more prominent.
    gaussian_conv = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=gaussian_kernel_size,
        padding='same',
        bias=False
    ).to(device)

    gaussian_k = gaussian_kernel_2d(gaussian_kernel_size, gaussian_sigma, in_channels=1, device=device)
    gaussian_conv.weight = nn.Parameter(gaussian_k)
    gaussian_conv.weight.requires_grad = False

    blurred_batch = gaussian_conv(conv_input)

    # Progressively downsample the smoothed error map using max pooling until 4x4.
    target_size = 4
    current_size = original_image_size
    num_pooling_steps = 0
    while current_size > target_size:
        current_size /= 2
        num_pooling_steps += 1
        if num_pooling_steps > 10: # Safety break for unusual image sizes
            break

    maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

    pooled_output = blurred_batch
    for _ in range(num_pooling_steps):
        pooled_output = maxpool_layer(pooled_output)
        if pooled_output.shape[-1] < 1: # Stop if dimension becomes zero
            break

    # Ensure final pooled output is exactly 4x4, using adaptive pooling as a fallback.
    if pooled_output.shape[-1] != target_size or pooled_output.shape[-2] != target_size:
        pooled_output = torch.nn.functional.adaptive_max_pool2d(pooled_output, (target_size, target_size))

    # Flatten the 4x4 map and extract the highest value (representing the most significant
    # localized error peak) for each image in the batch.
    flattened_output = pooled_output.view(N, -1)
    anomaly_scores = torch.max(flattened_output, dim=1).values

    return anomaly_scores