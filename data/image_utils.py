from PIL import Image
import numpy as np

def load_image(path):
  fmt = 'RGB'
  img = Image.open(path).convert(fmt)
  return img