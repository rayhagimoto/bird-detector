from PIL import Image
from PIL.Image import Image as PILImage
from io import BytesIO
import numpy as np
import csv

class AnomalyDetector:

  def __init__(self, config, s3):
    "Initializes the detector using a config (loaded from s3) and the s3 client itself."
    self.config = config
    self.s3 = s3
    # Load any additional files this AnomalyDetector instance needs.
    
  def predict(self, img: PILImage) -> bool:
    "Method that will return True if an anomaly is detected. False otherwise."
    return 
  
  def __call__(self, img: PILImage):
    return self.predict(img)
  
  # -- Common helper functions --
  def _load_image_from_s3(self, bucket, key):
    "Returns a PIL Image, with no transformations performed"
    response = self.s3.get_object(Bucket=bucket, key=key)
    img_data = response['Body'].read()
    img = Image.open(BytesIO(img_data)).convert('RGB')
    return img
  
  def _load_npy_from_s3(self, bucket, key):
    try:
      response = self.s3.get_object(Bucket=bucket, Key=key)
      return np.load(BytesIO(response['Body'].read()))
    except self.s3.exceptions.NoSuchKey:
      return None

  def _load_csv_from_s3(self, bucket, key):
    try:
      response = self.s3.get_object(Bucket=bucket, Key=key)
      lines = response['Body'].read().decode('utf-8').splitlines()
      return [float(row) for row in csv.reader(lines) if row][0]
    except self.s3.exceptions.NoSuchKey:
      return None
    
  def _save_npy_to_s3(self, array, bucket, key):
    buf = BytesIO()
    np.save(buf, array)
    buf.seek(0)
    self.s3.put_object(Bucket=bucket, Key=key, Body=buf)

  def _save_csv_to_s3(self, vals, bucket, key):
    "Assumes csv is a 1D column"
    csv_str = "\n".join(str(val) for val in vals)
    self.s3.put_object(Bucket=bucket, Key=key, Body=csv_str)

    

    


