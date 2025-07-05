import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def interactive_1d_latent_space(model, model_config:Dict, z_min=-4, z_max=4, z_initial=0.0):
    """
    Interactive 1D latent space exploration.
    
    Args:
        model: Autoencoder model with decoder method
        z_min: Minimum value for latent space exploration
        z_max: Maximum value for latent space exploration
        z_initial: Initial position in latent space
    """
    device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')

    IMAGE_SIZE = model_config.get('input_size', None)
    if not IMAGE_SIZE:
        raise ValueError("model_config must have `input_size` attribute")

    # Convert NumPy scalars to Python scalars to avoid matplotlib Slider issues
    z_min = float(z_min)
    z_max = float(z_max)
    z_initial = float(z_initial)

    # Set the range for the latent space axis
    LATENT_RANGE = (z_min, z_max)

    # Function to decode and display an image from a latent vector
    def decode_and_show(latent_val, ax):
        with torch.no_grad():
            # Create 1D latent vector (assuming 1D latent space)
            latent_tensor = torch.tensor([[latent_val]], dtype=torch.float32).to(device)
            decoded = model.decoder(latent_tensor).cpu().squeeze(0)
            if decoded.ndim == 1:
                decoded = decoded.view(3, IMAGE_SIZE, IMAGE_SIZE)  # adjust to your image shape
            img = decoded.permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f"Decoded image\nz=({latent_val:.2f})")
            ax.axis('off')

    # Create figure with subplots
    fig, (ax_img, ax_latent) = plt.subplots(1, 2, figsize=(10, 4))

    # Draw the 1D latent space
    ax_latent.set_xlim(LATENT_RANGE)
    ax_latent.set_ylim(-0.1, 0.1)  # Small range for visual effect
    ax_latent.set_xlabel('Latent dimension')
    ax_latent.set_ylabel('')  # No y-label for 1D
    ax_latent.set_title('1D Latent Space')
    ax_latent.grid(True, alpha=0.3)
    ax_latent.set_yticks([])  # Hide y-axis ticks

    # Initial latent value
    z = z_initial
    point_handle = ax_latent.plot(z, 0, 'ro', markersize=10)[0]
    
    # Add a vertical line to show the current position
    line_handle = ax_latent.axvline(x=z, color='red', alpha=0.5, linestyle='--')

    def update_display(latent_val):
        """Update the display with new latent value."""
        ax_img.clear()
        decode_and_show(latent_val, ax_img)
        point_handle.set_data([latent_val], [0])
        line_handle.set_xdata([latent_val, latent_val])
        fig.canvas.draw_idle()

    def onclick(event):
        """Handle mouse clicks on the latent space."""
        if event.inaxes == ax_latent:
            z = event.xdata
            # Clamp to range
            z = np.clip(z, LATENT_RANGE[0], LATENT_RANGE[1])
            update_display(z)

    def onmove(event):
        """Handle mouse drag on the latent space."""
        if event.inaxes == ax_latent and event.button == 1:  # Left mouse button
            z = event.xdata
            # Clamp to range
            z = np.clip(z, LATENT_RANGE[0], LATENT_RANGE[1])
            update_display(z)

    # Add a slider for precise control
    from matplotlib.widgets import Slider
    
    # Create axes for the slider
    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        slider_ax, 'Latent Value', 
        LATENT_RANGE[0], LATENT_RANGE[1], 
        valinit=z_initial, valstep=0.1
    )
    
    def slider_update(val):
        """Update display when slider changes."""
        update_display(val)
    
    slider.on_changed(slider_update)

    # Initial plot
    update_display(z)
    
    # Connect events
    fig.canvas.mpl_connect('motion_notify_event', onmove)
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.tight_layout()
    plt.show()

def interactive_2d_latent_space(model, model_config: Dict, boundaries: List[float], latent_space_vals: np.ndarray = None, hist_kwargs=None):
    """
    Interactive 2D latent space exploration.
    
    Args:
        model: Autoencoder model with decoder method
        model_config: Model configuration dictionary
        boundaries: List of [x_min, x_max, y_min, y_max] for latent space bounds
        latent_space_vals: Optional, (N, 2) array of latent samples to plot as 2D histogram background
    """
    X_MIN, X_MAX, Y_MIN, Y_MAX = boundaries

    device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
    
    IMAGE_SIZE = model_config.get('input_size', None)
    if not IMAGE_SIZE:
        raise ValueError("model_config must have `input_size` attribute")

    # Set the range for the latent space axes
    LATENT_X_RANGE = (X_MIN, X_MAX)
    LATENT_Y_RANGE = (Y_MIN, Y_MAX)
    
    _hist_kwargs = {'cmap': 'gray_r', 'bins': 20, 'alpha': 0.5}
    if hist_kwargs:
        _hist_kwargs.update(hist_kwargs)
    
    _hist_kwargs['range'] = [LATENT_X_RANGE, LATENT_Y_RANGE]
    

    # Function to decode and display an image from a latent vector
    def decode_and_show(latent_vec, ax):
        with torch.no_grad():
            latent_tensor = torch.tensor(latent_vec, dtype=torch.float32).unsqueeze(0).to(device)
            decoded = model.decoder(latent_tensor).cpu().squeeze(0)
            if decoded.ndim == 1:
                decoded = decoded.view(3, IMAGE_SIZE, IMAGE_SIZE)  # Use IMAGE_SIZE from config
            img = decoded.permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f"Decoded image\nz=({latent_vec[0]:.2f}, {latent_vec[1]:.2f})")
            ax.axis('off')

    fig, (ax_img, ax_latent) = plt.subplots(1, 2, figsize=(10, 4))

    # Draw the latent space grid
    ax_latent.set_xlim(LATENT_X_RANGE)
    ax_latent.set_ylim(LATENT_Y_RANGE)
    ax_latent.set_xlabel('Latent dim 1')
    ax_latent.set_ylabel('Latent dim 2')
    ax_latent.set_title('2D Latent Space')
    ax_latent.grid(True, alpha=0.3)

    # Optionally plot 2D histogram of latent samples as background
    if latent_space_vals is not None and latent_space_vals.shape[1] == 2:
        h = ax_latent.hist2d(latent_space_vals[:, 0], latent_space_vals[:, 1], **_hist_kwargs)
        plt.colorbar(h[3], ax=ax_latent, label='Count', fraction=0.046, pad=0.04)

    # Initial latent vector (center of the space)
    z = [0.5 * (X_MIN + X_MAX), 0.5 * (Y_MIN + Y_MAX)]
    point_handle = ax_latent.plot(z[0], z[1], 'ro', markersize=10)[0]

    def onclick(event):
        if event.inaxes == ax_latent:
            z[0], z[1] = event.xdata, event.ydata
            ax_img.clear()
            decode_and_show(z, ax_img)
            point_handle.set_data([z[0]], [z[1]])
            fig.canvas.draw_idle()

    def onmove(event):
        if event.inaxes == ax_latent and event.button == 1:  # Only respond if left mouse button is pressed
            z[0], z[1] = event.xdata, event.ydata
            ax_img.clear()
            decode_and_show(z, ax_img)
            point_handle.set_data([z[0]], [z[1]])
            fig.canvas.draw_idle()

    # Initial plot
    decode_and_show(z, ax_img)
    fig.canvas.mpl_connect('motion_notify_event', onmove)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()
