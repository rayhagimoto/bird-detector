import numpy as np

def otsu_method(x, n_thresholds=100):
  "Computes threshold that minimizes variance of binary classes."
  x = np.atleast_1d(x)
  x_min = x.min()
  x_max = x.max()
  best_var = float('inf')
  best_t = None

  thresholds = np.linspace(x_min, x_max, n_thresholds)
  for t in thresholds:
    mask = x < t
    if mask.any(): 
      var1 = np.var(x[mask])
      var2 = np.var(x[~mask])

      p1 = np.mean(mask)
      p2 = 1 - p1
      wgt_var = p1 * var1 + p2 * var2
      if wgt_var < best_var:
        best_var = wgt_var
        best_t = t

  return best_t