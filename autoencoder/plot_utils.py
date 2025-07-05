# Plot utils for autoencoder pipeline

import matplotlib.pyplot as plt
import numpy as np

# Distribution of the errors
def plot_mse_scores_hist(mse_scores, anomaly_threshold, percentile, **kwargs):

  # Visualize the MSE distribution
  plt.figure(figsize=(12, 4))

  plt.subplot(1, 2, 1)
  plt.hist(mse_scores, bins=50, alpha=0.7, color='blue', density=True)
  plt.axvline(anomaly_threshold, color='red', linestyle='--', 
            label=f'{float(percentile):.1f}th percentile ({anomaly_threshold:.4f})')
  plt.xlabel('MSE Score')
  plt.ylabel('Density')
  plt.title('MSE Score Distribution (Full Dataset)')
  plt.legend()
  plt.grid(True, alpha=0.3)

  plt.subplot(1, 2, 2)
  plt.hist(np.log10(mse_scores), bins=50, alpha=0.7, color='green', density=True)
  plt.axvline(np.log10(anomaly_threshold), color='red', linestyle='--', 
            label=f'{float(percentile):.1f}th percentile (log10: {np.log10(anomaly_threshold):.4f})')
  plt.xlabel('log10(MSE Score)')
  plt.ylabel('Density')
  plt.title('Log10 MSE Score Distribution')
  plt.legend()
  plt.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.show()