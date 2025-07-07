import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from .anomaly_detector import AnomalyDetector

class OpenCVDetector(AnomalyDetector):
    def __init__(self, config, s3):
        super().__init__(config, s3)
        self.bucket = config.get('bucket_name', 'your-s3-bucket')
        self.state_folder = config.get('state_folder', 'py')
        self.ema_filename = config.get('ema_filename', 'ema.npy')
        self.ema_key = f'{self.state_folder}/{self.ema_filename}'
        self.image_size = tuple(config.get('image_size', [128, 128]))
        self.alpha = config.get('alpha', 0.05)
        self.threshold = config.get('threshold', 0.15)
        self.blur_sigma = config.get('blur_sigma', 0.5)
        self.min_area_frac = config.get('min_area_frac', 0.0015)
        self.max_area_frac = config.get('max_area_frac', 0.05)
        self.exclude_bottom = config.get('exclude_bottom', True)
        self.aspect_ratio_max = config.get('aspect_ratio_max', 2.0)
        self.preferred_vertical_range = tuple(config.get('preferred_vertical_range', [0.15, 0.85]))
        self.min_contrast = config.get('min_contrast', 10)

    def _compute_luminance(self, x):
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def _score(self, contour, diff_map):
        area = cv2.contourArea(contour)
        area_frac = area / (diff_map.shape[0] * diff_map.shape[1])
        mask = np.zeros_like(diff_map)
        cv2.drawContours(mask, [contour], 0, 1, -1)
        vals = diff_map[mask.astype(bool)]
        if len(vals) == 0:
            return float('-inf')
        diff_lum = np.mean(vals**2)
        return diff_lum * np.sqrt(area_frac)

    def _get_best_contour(self, diff_map, binary_map):
        try:
            contours, _ = cv2.findContours(
                binary_map.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
        except Exception as e:
            print(f"Error finding contours: {str(e)}")
            return None
        if not contours:
            return None
        best_contour = None
        best_score = float('-inf')
        total_area = diff_map.shape[0] * diff_map.shape[1]
        for contour in contours:
            try:
                area = cv2.contourArea(contour)
                area_frac = area / total_area
                if area_frac < self.min_area_frac:
                    continue
                if area_frac > self.max_area_frac:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else float('inf')
                if aspect_ratio > self.aspect_ratio_max:
                    continue
                center_y = (y + h/2) / diff_map.shape[0]
                if not (self.preferred_vertical_range[0] <= center_y <= self.preferred_vertical_range[1]):
                    continue
                if self.exclude_bottom and y + h > diff_map.shape[0] * 0.67:
                    continue
                mask = np.zeros_like(diff_map)
                cv2.drawContours(mask, [contour], 0, 1, -1)
                vals = diff_map[mask.astype(bool)]
                if len(vals) == 0:
                    continue
                contrast = (vals.max() - vals.min()) * 255
                if contrast < self.min_contrast:
                    continue
                score = self._score(contour, diff_map)
                if score > best_score:
                    best_score = score
                    best_contour = contour
            except Exception as e:
                print(f"Error processing contour: {str(e)}")
                continue
        return best_contour

    def predict(self, img: Image.Image) -> bool:
        # Convert PIL Image to numpy array and resize
        img_np = np.asarray(img.convert("RGB").resize(self.image_size, Image.BICUBIC), dtype=np.uint8)
        img_np = img_np.astype(np.float32) / 255.0
        # Load EMA from S3
        ema = super()._load_npy_from_s3(self.bucket, self.ema_key)
        if ema is None:
            ema = img_np.copy()
        # Update EMA
        ema = (1 - self.alpha) * ema + self.alpha * img_np
        # Save updated EMA
        super()._save_npy_to_s3(ema, self.bucket, self.ema_key)
        # Anomaly detection
        curr_lum = self._compute_luminance(img_np)
        ema_lum = self._compute_luminance(ema)
        diff_map = np.abs(curr_lum - ema_lum)
        blurred_diff = cv2.GaussianBlur(diff_map, (0, 0), self.blur_sigma)
        binary_map = (blurred_diff > self.threshold).astype(np.float32)
        result = self._get_best_contour(diff_map, binary_map)
        return result is not None 