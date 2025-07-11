# Routines for importing and processing raw images. 

import rawpy
import numpy as np
from pathlib import Path
from typing import Union


def load_arw_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Open an ARW raw image file and return it as a numpy array.
    
    Args:
        image_path: Path to the ARW raw image file
        
    Returns:
        numpy.ndarray: Image as a (height, width, channels) numpy array
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        rawpy.LibRawError: If there's an error reading the raw file
    """
    # Convert to Path object for better handling
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Open the raw image
    with rawpy.imread(str(image_path)) as raw:
        # Convert to RGB using the camera's color profile
        rgb = raw.postprocess(
            use_camera_wb=True,  # Use camera white balance
            half_size=False,     # Full resolution
            no_auto_bright=False, # Allow auto brightness adjustment
            output_bps=8         # 8-bit output
        )
        
        return rgb 
    
def arw_to_jpg(raw_img: np.ndarray, quality: int = 95) -> bytes:
    """
    Convert a raw image array to JPG format in memory.
    
    Args:
        raw_img: Raw image as numpy array (from load_arw_image)
        quality: JPG quality (1-100, higher is better quality)
        
    Returns:
        bytes: JPG image data as bytes
        
    Raises:
        ValueError: If quality is not between 1 and 100
    """
    import cv2
    
    if not 1 <= quality <= 100:
        raise ValueError("Quality must be between 1 and 100")
    
    # Ensure image is in the correct format for cv2
    if raw_img.dtype != np.uint8:
        # Normalize to 0-255 range if needed
        if raw_img.max() <= 1.0:
            raw_img = (raw_img * 255).astype(np.uint8)
        else:
            raw_img = raw_img.astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV (if needed)
    if len(raw_img.shape) == 3 and raw_img.shape[2] == 3:
        # OpenCV expects BGR format
        img_bgr = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = raw_img
    
    # Encode to JPG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, jpg_data = cv2.imencode('.jpg', img_bgr, encode_params)
    
    if not success:
        raise RuntimeError("Failed to encode image to JPG format")
    
    return jpg_data.tobytes()