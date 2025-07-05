import numpy as np
import colorsys

def rgb_to_lab(rgb_image):
  if rgb_image.dtype != np.uint8:
    raise ValueError("Input image must be of type uint8.")
  if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
    raise ValueError("Input image must be a 3-channel RGB image (H, W, 3).")

  rgb_normalized = rgb_image.astype(np.float64) / 255.0

  h, w, c = rgb_normalized.shape

  lab_image = np.empty((h, w, 3), dtype=np.float64)

  Xn, Yn, Zn = 95.047, 100.000, 108.883

  def _srgb_to_linear(value):
    if value <= 0.04045:
      return value / 12.92
    else:
      return ((value + 0.055) / 1.055) ** 2.4

  def _linear_rgb_to_xyz(R, G, B):
    X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B
    Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
    Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B
    return X * 100, Y * 100, Z * 100

  def _f(t):
    if t > (6/29)**3:
      return t**(1/3)
    else:
      return (1/3) * ((29/6)**2) * t + 4/29

  def _xyz_to_lab(X, Y, Z):
    Xr = X / Xn
    Yr = Y / Yn
    Zr = Z / Zn

    fx, fy, fz = _f(Xr), _f(Yr), _f(Zr)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b

  for i in range(h):
    for j in range(w):
      r_srgb, g_srgb, b_srgb = rgb_normalized[i, j, :]

      r_linear = _srgb_to_linear(r_srgb)
      g_linear = _srgb_to_linear(g_srgb)
      b_linear = _srgb_to_linear(b_srgb)

      X, Y, Z = _linear_rgb_to_xyz(r_linear, g_linear, b_linear)

      L, a, b = _xyz_to_lab(X, Y, Z)

      lab_image[i, j, 0] = L
      lab_image[i, j, 1] = a
      lab_image[i, j, 2] = b

  return lab_image