import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d


def gaussian_kernel_2d_numpy(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Create a 2D Gaussian kernel for numpy/scipy operations.
    
    Args:
        kernel_size: Size of the kernel (should be odd)
        sigma: Standard deviation of the Gaussian
        
    Returns:
        np.ndarray: 2D Gaussian kernel
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size
    
    # Create coordinate grids
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    
    # Create Gaussian kernel
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Normalize
    kernel = kernel / np.sum(kernel)
    
    return kernel


def localized_reconstruction_score_numpy(
    diff: np.ndarray,
    gaussian_kernel_size: int = 10,
    gaussian_sigma: float = 1.0
) -> float:
    """
    Calculates an anomaly score by identifying the peak intensity within a downsampled,
    smoothed reconstruction error map. Non-PyTorch version using numpy/scipy.

    This score is derived from:
    1.  Averaging input `diff` channels to a single grayscale error map.
    2.  Applying Gaussian smoothing to the error map.
    3.  Sequentially max-pooling the smoothed error map down to a 4x4 spatial resolution.
    4.  Flattening the 4x4 map and extracting the maximum value as the anomaly score.
    
    Args:
        diff: Difference image as numpy array of shape (H, W, C) or (C, H, W)
        gaussian_kernel_size: Size of Gaussian kernel for smoothing
        gaussian_sigma: Standard deviation for Gaussian smoothing
        
    Returns:
        float: Anomaly score for the single image
    """
    # Handle different input formats
    if diff.ndim == 3:
        if diff.shape[0] <= 3:  # (C, H, W) format
            C, H, W = diff.shape
            # Convert to (H, W, C) for easier processing
            diff = np.transpose(diff, (1, 2, 0))
        else:  # (H, W, C) format
            H, W, C = diff.shape
    else:
        raise ValueError(f"Expected 3D array, got {diff.ndim}D")
    
    # Convert multi-channel difference map to a single-channel grayscale error map
    if C >= 2:
        processed_diff_for_blur = 1.0 / 3.0 * (diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2])
    else:
        processed_diff_for_blur = diff.squeeze()
    
    # Apply Gaussian smoothing to blur the error map
    gaussian_kernel = gaussian_kernel_2d_numpy(gaussian_kernel_size, gaussian_sigma)
    blurred = convolve2d(
        processed_diff_for_blur, 
        gaussian_kernel, 
        mode='same', 
        boundary='symm'
    )
    
    # Progressively downsample the smoothed error map using max pooling until 4x4
    target_size = 4
    current_size = H  # Assuming square images
    num_pooling_steps = 0
    while current_size > target_size:
        current_size /= 2
        num_pooling_steps += 1
        if num_pooling_steps > 10:  # Safety break for unusual image sizes
            break
    
    # Apply max pooling
    pooled_output = blurred
    for _ in range(num_pooling_steps):
        # Manual max pooling with stride 2
        h, w = pooled_output.shape
        new_h, new_w = h // 2, w // 2
        
        # Create output array
        new_pooled = np.zeros((new_h, new_w))
        
        for y in range(new_h):
            for x in range(new_w):
                # Extract 2x2 patch and take max
                patch = pooled_output[y*2:(y+1)*2, x*2:(x+1)*2]
                new_pooled[y, x] = np.max(patch)
        
        pooled_output = new_pooled
        
        # Stop if dimensions become too small
        if pooled_output.shape[0] < 1 or pooled_output.shape[1] < 1:
            break
    
    # Ensure final pooled output is exactly 4x4 using adaptive pooling
    if pooled_output.shape[0] != target_size or pooled_output.shape[1] != target_size:
        # Manual adaptive max pooling to 4x4
        h, w = pooled_output.shape
        new_pooled = np.zeros((target_size, target_size))
        
        for y in range(target_size):
            for x in range(target_size):
                # Calculate the region this output pixel should cover
                y_start = int(y * h / target_size)
                y_end = int((y + 1) * h / target_size)
                x_start = int(x * w / target_size)
                x_end = int((x + 1) * w / target_size)
                
                # Extract region and take max
                region = pooled_output[y_start:y_end, x_start:x_end]
                new_pooled[y, x] = np.max(region)
        
        pooled_output = new_pooled
    
    # Flatten the 4x4 map and extract the maximum value
    flattened_output = pooled_output.flatten()
    anomaly_score = np.max(flattened_output)
    
    return anomaly_score
