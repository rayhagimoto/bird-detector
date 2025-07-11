import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from .anomaly_detector import AnomalyDetector
import pickle

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
        self.min_observations = config.get('min_observations', 10)
        self.counter_key = f'{self.state_folder}/opencv_counter.txt'

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

    def _find_best_score_and_contour(self, diff_map, binary_map):
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
            print(f"Error finding contours: {str(e)}")
        return best_score, best_contour

    def _load_state_dict(self):
        key = f"{self.state_folder}/state_dict.pkl"
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            state_dict = pickle.load(BytesIO(response['Body'].read()))
            return state_dict
        except Exception:
            return None

    def _save_state_dict(self, state_dict):
        key = f"{self.state_folder}/state_dict.pkl"
        buf = BytesIO()
        pickle.dump(state_dict, buf)
        buf.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())

    def predict(self, img: Image.Image, filename=None) -> bool:
        # Convert PIL Image to numpy array and resize
        img_np = np.asarray(img.convert("RGB").resize(self.image_size, Image.BICUBIC), dtype=np.uint8)
        img_np = img_np.astype(np.float32) / 255.0
        # Load state_dict from S3
        state_dict = self._load_state_dict()
        if state_dict is None:
            ema = img_np.copy()
            count = 1
            scores = []
            binary_maps = []
            anomalies = []
        else:
            ema = state_dict.get('ema', img_np.copy())
            count = state_dict.get('counter', 0) + 1
            scores = state_dict.get('scores', [])
            binary_maps = state_dict.get('binary_maps', [])
            anomalies = state_dict.get('anomalies', [])
        # Update EMA
        ema = (1 - self.alpha) * ema + self.alpha * img_np
        # Anomaly detection
        curr_lum = self._compute_luminance(img_np)
        ema_lum = self._compute_luminance(ema)
        diff_map = np.abs(curr_lum - ema_lum)
        blurred_diff = cv2.GaussianBlur(diff_map, (0, 0), self.blur_sigma)
        binary_map = (blurred_diff > self.threshold).astype(np.float32)
        # Save a copy of the binary map for this image
        binary_maps.append(binary_map.copy())
        # Find best score and contour for this image
        best_score, best_contour = self._find_best_score_and_contour(diff_map, binary_map)
        scores.append(best_score)
        print(f"[OpenCVDetector] Processing image #{count}")
        if count < self.min_observations:
            print(f"[OpenCVDetector] Not enough observations yet: {count}/{self.min_observations}")
            print(f"[OpenCVDetector] No anomaly detected for this image. Reason: Not enough observations.")
            # Save updated state_dict
            state_dict = {'ema': ema, 'counter': count, 'scores': scores, 'binary_maps': binary_maps, 'anomalies': anomalies}
            self._save_state_dict(state_dict)
            return False
        if best_score == float('-inf'):
            print(f"[OpenCVDetector] No anomaly detected for image #{count}. Reason: No valid contours found.")
            # Save updated state_dict
            state_dict = {'ema': ema, 'counter': count, 'scores': scores, 'binary_maps': binary_maps, 'anomalies': anomalies}
            self._save_state_dict(state_dict)
            return False
        print(f"[OpenCVDetector] Anomaly detected for image #{count}! Best score: {best_score:.4f} (threshold: {self.threshold})")
        # Append anomaly filename if provided
        if filename is not None:
            anomalies.append(filename)
        # Save updated state_dict
        state_dict = {'ema': ema, 'counter': count, 'scores': scores, 'binary_maps': binary_maps, 'anomalies': anomalies}
        self._save_state_dict(state_dict)
        return True 