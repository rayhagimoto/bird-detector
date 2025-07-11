import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from .anomaly_detector import AnomalyDetector
import pickle
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class OpenCVConfig:
    """Configuration for OpenCV anomaly detection."""
    image_size: Tuple[int, int] = (128, 128)
    alpha: float = 0.05
    threshold: float = 0.15
    blur_sigma: float = 0.5
    min_area_frac: float = 0.0015
    max_area_frac: float = 0.05
    exclude_bottom: bool = True
    aspect_ratio_max: float = 2.0
    preferred_vertical_range: Tuple[float, float] = (0.15, 0.85)
    min_contrast: float = 10.0
    min_observations: int = 10

@dataclass
class DetectionState:
    """State for anomaly detection."""
    ema: np.ndarray
    counter: int
    scores: List[float]
    binary_maps: List[np.ndarray]
    anomalies: List[str]

@dataclass
class DetectionResult:
    """Result of anomaly detection."""
    is_anomaly: bool
    score: float
    contour: Optional[np.ndarray]
    diff_map: np.ndarray
    binary_map: np.ndarray
    reason: str

def compute_luminance(image: np.ndarray) -> np.ndarray:
    """
    Compute luminance from RGB image using standard coefficients.
    
    Args:
        image: RGB image array of shape (H, W, 3) with values in [0, 1]
        
    Returns:
        Luminance array of shape (H, W) with values in [0, 1]
    """
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def score_contour(contour: np.ndarray, diff_map: np.ndarray) -> float:
    """
    Score a contour based on area and luminance difference.
    
    Args:
        contour: OpenCV contour
        diff_map: Difference map between current and background
        
    Returns:
        Score for the contour (higher is more anomalous)
    """
    area = cv2.contourArea(contour)
    area_frac = area / (diff_map.shape[0] * diff_map.shape[1])
    mask = np.zeros_like(diff_map)
    cv2.drawContours(mask, [contour], 0, 1, -1)
    vals = diff_map[mask.astype(bool)]
    if len(vals) == 0:
        return float('-inf')
    diff_lum = np.mean(vals**2)
    return diff_lum * np.sqrt(area_frac)

def find_best_contour(
    diff_map: np.ndarray, 
    binary_map: np.ndarray, 
    config: OpenCVConfig
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Find the best scoring contour in the binary map.
    
    Args:
        diff_map: Difference map between current and background
        binary_map: Binary map of anomalous regions
        config: Configuration parameters
        
    Returns:
        Tuple of (best_score, best_contour)
    """
    best_contour = None
    best_score = float('-inf')
    
    try:
        contours, _ = cv2.findContours(
            binary_map.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            area = cv2.contourArea(contour)
            area_frac = area / (diff_map.shape[0] * diff_map.shape[1])
            
            # Filter by area
            if area_frac < config.min_area_frac or area_frac > config.max_area_frac:
                continue
                
            # Filter by aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else float('inf')
            if aspect_ratio > config.aspect_ratio_max:
                continue
                
            # Filter by vertical position
            center_y = (y + h/2) / diff_map.shape[0]
            if not (config.preferred_vertical_range[0] <= center_y <= config.preferred_vertical_range[1]):
                continue
                
            # Exclude bottom region if configured
            if config.exclude_bottom and y + h > diff_map.shape[0] * 0.67:
                continue
                
            # Filter by contrast
            mask = np.zeros_like(diff_map)
            cv2.drawContours(mask, [contour], 0, 1, -1)
            vals = diff_map[mask.astype(bool)]
            if len(vals) == 0:
                continue
            contrast = (vals.max() - vals.min()) * 255
            if contrast < config.min_contrast:
                continue
                
            score = score_contour(contour, diff_map)
            if score > best_score:
                best_score = score
                best_contour = contour
                
    except Exception as e:
        print(f"Error finding contours: {str(e)}")
        
    return best_score, best_contour

def preprocess_image(image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    img_np = np.asarray(
        image.convert("RGB").resize(target_size, Image.BICUBIC), 
        dtype=np.uint8
    )
    return img_np.astype(np.float32) / 255.0

def update_ema(current_image: np.ndarray, ema: np.ndarray, alpha: float) -> np.ndarray:
    return (1 - alpha) * ema + alpha * current_image

def detect_anomaly(
    image: np.ndarray,
    ema: np.ndarray,
    config: OpenCVConfig
) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """
    Detect anomalies in an image compared to background model.
    
    Args:
        image: Current image array of shape (H, W, 3) with values in [0, 1]
        ema: Background model (exponential moving average)
        config: Configuration parameters
        
    Returns:
        Tuple of (diff_map, binary_map, best_score, best_contour)
    """
    # Compute luminance difference
    curr_lum = compute_luminance(image)
    ema_lum = compute_luminance(ema)
    diff_map = np.abs(curr_lum - ema_lum)
    
    # Apply Gaussian blur and threshold
    blurred_diff = cv2.GaussianBlur(diff_map, (0, 0), config.blur_sigma)
    binary_map = (blurred_diff > config.threshold).astype(np.float32)
    
    # Find best contour
    best_score, best_contour = find_best_contour(diff_map, binary_map, config)
    
    return diff_map, binary_map, best_score, best_contour

def process_image(
    image: Image.Image,
    state: Optional[DetectionState],
    config: OpenCVConfig,
    filename: Optional[str] = None
) -> Tuple[DetectionResult, DetectionState]:
    """
    Process a single image for anomaly detection using functional approach.
    
    Args:
        image: PIL Image to process
        state: Current detection state (None for first image)
        config: Configuration parameters
        filename: Optional filename for tracking anomalies
        
    Returns:
        Tuple of (detection_result, updated_state)
    """
    # Preprocess image
    img_np = preprocess_image(image, config.image_size)
    
    # Initialize or update state
    if state is None:
        ema = img_np.copy()
        counter = 1
        scores = []
        binary_maps = []
        anomalies = []
    else:
        ema = update_ema(img_np, state.ema, config.alpha)
        counter = state.counter + 1
        scores = state.scores.copy()
        binary_maps = state.binary_maps.copy()
        anomalies = state.anomalies.copy()
    
    # Detect anomalies
    diff_map, binary_map, best_score, best_contour = detect_anomaly(img_np, ema, config)
    
    # Update tracking
    scores.append(best_score)
    binary_maps.append(binary_map.copy())
    
    # Determine if anomaly
    is_anomaly = False
    reason = ""
    
    if counter < config.min_observations:
        reason = f"Not enough observations yet: {counter}/{config.min_observations}"
    elif best_score == float('-inf'):
        reason = "No valid contours found"
    else:
        is_anomaly = True
        reason = f"Anomaly detected! Score: {best_score:.4f}"
        if filename is not None:
            anomalies.append(filename)
    
    # Create result and updated state
    result = DetectionResult(
        is_anomaly=is_anomaly,
        score=best_score,
        contour=best_contour,
        diff_map=diff_map,
        binary_map=binary_map,
        reason=reason
    )
    
    updated_state = DetectionState(
        ema=ema,
        counter=counter,
        scores=scores,
        binary_maps=binary_maps,
        anomalies=anomalies
    )
    
    return result, updated_state

def process_image_batch_functional(
    images: List[Image.Image],
    config: OpenCVConfig,
    filenames: Optional[List[str]] = None
) -> Tuple[List[DetectionResult], DetectionState]:
    
    state = None
    results = []
    
    for i, image in enumerate(images):
        filename = filenames[i] if filenames and i < len(filenames) else None
        result, state = process_image(image, state, config, filename)
        results.append(result)
    
    return results, state

class OpenCVDetector(AnomalyDetector):
    """S3-backed OpenCV anomaly detector (original implementation)."""
    
    def __init__(self, config, s3):
        super().__init__(config, s3)
        self.bucket = config.get('bucket_name', 'your-s3-bucket')
        self.state_folder = config.get('state_folder', 'py')
        self.ema_filename = config.get('ema_filename', 'ema.npy')
        self.ema_key = f'{self.state_folder}/{self.ema_filename}'
        
        # Create OpenCVConfig from dict config
        self.opencv_config = OpenCVConfig(
            image_size=tuple(config.get('image_size', [128, 128])),
            alpha=config.get('alpha', 0.05),
            threshold=config.get('threshold', 0.15),
            blur_sigma=config.get('blur_sigma', 0.5),
            min_area_frac=config.get('min_area_frac', 0.0015),
            max_area_frac=config.get('max_area_frac', 0.05),
            exclude_bottom=config.get('exclude_bottom', True),
            aspect_ratio_max=config.get('aspect_ratio_max', 2.0),
            preferred_vertical_range=tuple(config.get('preferred_vertical_range', [0.15, 0.85])),
            min_contrast=config.get('min_contrast', 10),
            min_observations=config.get('min_observations', 10)
        )
        
        self.counter_key = f'{self.state_folder}/opencv_counter.txt'

    def _load_state_dict(self) -> Optional[Dict[str, Any]]:
        """Load state dictionary from S3."""
        key = f"{self.state_folder}/state_dict.pkl"
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            state_dict = pickle.load(BytesIO(response['Body'].read()))
            return state_dict
        except Exception:
            return None

    def _save_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Save state dictionary to S3."""
        key = f"{self.state_folder}/state_dict.pkl"
        buf = BytesIO()
        pickle.dump(state_dict, buf)
        buf.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())

    def _load_ema_from_s3(self) -> Optional[np.ndarray]:
        """Load EMA array from S3 using base class helper."""
        return self._load_npy_from_s3(self.bucket, self.ema_key)

    def _save_ema_to_s3(self, ema: np.ndarray) -> None:
        """Save EMA array to S3 using base class helper."""
        self._save_npy_to_s3(ema, self.bucket, self.ema_key)

    def _load_scores_from_s3(self) -> Optional[List[float]]:
        """Load scores from S3 using base class helper."""
        scores_key = f"{self.state_folder}/scores.csv"
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=scores_key)
            lines = response['Body'].read().decode('utf-8').splitlines()
            return [float(line.strip()) for line in lines if line.strip()]
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception:
            return None

    def _save_scores_to_s3(self, scores: List[float]) -> None:
        """Save scores to S3 using base class helper."""
        scores_key = f"{self.state_folder}/scores.csv"
        self._save_csv_to_s3(scores, self.bucket, scores_key)

    def _state_to_dict(self, state: DetectionState) -> Dict[str, Any]:
        """Convert DetectionState to dictionary for S3 storage."""
        return {
            'ema': state.ema,
            'counter': state.counter,
            'scores': state.scores,
            'binary_maps': state.binary_maps,
            'anomalies': state.anomalies
        }

    def _dict_to_state(self, state_dict: Dict[str, Any]) -> DetectionState:
        """Convert dictionary from S3 to DetectionState."""
        return DetectionState(
            ema=state_dict.get('ema'),
            counter=state_dict.get('counter', 0),
            scores=state_dict.get('scores', []),
            binary_maps=state_dict.get('binary_maps', []),
            anomalies=state_dict.get('anomalies', [])
        )

    def predict(self, img: Image.Image, filename: Optional[str] = None) -> bool:

        # Load state from S3
        state_dict = self._load_state_dict()
        state = self._dict_to_state(state_dict) if state_dict else None
        
        # Process image using functional approach
        result, updated_state = process_image(img, state, self.opencv_config, filename)
        
        # Save updated state to S3
        self._save_state_dict(self._state_to_dict(updated_state))
        
        # Print logging information
        print(f"[OpenCVDetector] Processing image #{updated_state.counter}")
        print(f"[OpenCVDetector] {result.reason}")
        
        return result.is_anomaly 