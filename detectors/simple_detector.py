import numpy as np

from .anomaly_detector import AnomalyDetector

class SimpleDetector(AnomalyDetector):
    
    def __init__(self, config, s3):
      super().__init__(config, s3)

      # -- File names and S3 directory config --
      self.bucket = config.get('bucket_name', 'axiondm-photos')
      self.state_folder = config.get('state_folder', 'py')
      self.ema_filename = config.get('ema_filename', 'ema.npy')
      self.scores_filename = config.get('scores_filename', 'losses.csv')

      self.ema_key = f'{self.state_folder}/{self.ema_filename}'
      self.scores_key = f'{self.state_folder}/{self.scores_filename}'

      # -- Current image config --
      # Must be a list of ints with length 2. 
      # Be careful because this method won't check length or dtype.
      self.image_size = config.get('image_size', [64, 64]) 

      # -- Files needed for inference --      
      # These should be done concurrently.
      self.ema = super()._load_npy_from_s3(self.bucket, self.ema_key)
      self.scores = super()._load_csv_from_s3(self.bucket, self.scores_key)

      # -- Anomaly detection config -- 
      self.percentile = config.get('percentile', 98)
      self.alpha = config.get('alpha', 0.05)
      self.min_observations = config.get('min_observations', 25)

    def transform_img(self, img):
      "Loads image from S3 and performs transformations"
      # Perform transformation on the image
      img = img.resize(self.image_size)
      return np.asarray(img) / 255.0
    
    def predict(self, img) -> bool:
      "Returns true if img contains an anomaly, false if not."
      
      img_np = self.transform_img(img)
      img_flat = img_np.transpose(2, 0, 1).reshape(-1)
      if self.ema is None:
        self.ema = img_flat
      
      self.ema = (1 - self.alpha) * self.ema + self.alpha * img_flat
      score = np.log10(np.mean((img_flat - self.ema)**2))
      self.scores.append(score)

      # Save updated state
      super()._save_npy_to_s3(self.ema, self.bucket, self.ema_key)
      super()._save_csv_to_s3(self.scores, self.bucket, self.scores_key)

      # Anomaly logic
      threshold = None
      if len(self.scores) > self.min_observations:
        threshold = np.percentile(self.scores, self.percentile)
      
      return len(self.scores) > self.min_observations and score >= threshold
          