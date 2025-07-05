from tqdm import tqdm
import torch
from time import perf_counter
from .torch_utils import get_loss_function

def train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, training_config, 
                metrics_to_track=None, device=None, patience=5):
    """
    Train model with multiple metrics tracking and early stopping.
    
    Args:
        metrics_to_track: List of metric names to track (e.g., ['mse', 'mae', 'log_mse'])
        patience: Number of epochs to wait for improvement before early stopping
    """
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Default metrics if none specified
    if metrics_to_track is None:
        metrics_to_track = ['loss']
    
    # Define metric functions
    metric_functions = {
        'mse': lambda pred, target: torch.mean((pred - target) ** 2),
        'mae': lambda pred, target: torch.mean(torch.abs(pred - target)),
        'log_mse': get_loss_function('log_mse'),
    }
    
    print("Starting training...")
    t1 = perf_counter()
    num_epochs = training_config['epochs']
    
    # Initialize tracking dictionaries
    training_metrics = {metric: [] for metric in metrics_to_track}
    validation_metrics = {metric: [] for metric in metrics_to_track}
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0):
        # Training phase
        model.train()
        epoch_metrics = {metric: 0.0 for metric in metrics_to_track}
        
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Train", position=1, leave=False) as train_bar:
            for batch_idx, (data, filenames) in train_bar:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                
                # Calculate all metrics
                for metric in metrics_to_track:
                    if metric == 'loss':
                        metric_value = loss_fn(output, data)
                    else:
                        metric_value = metric_functions[metric](output, data)
                    epoch_metrics[metric] += metric_value.item()
                
                # Use loss for backprop
                loss = loss_fn(output, data)
                loss.backward()
                optimizer.step()
                
                # Update progress bar with all metrics
                postfix = {metric: f"{epoch_metrics[metric]/(batch_idx+1):.6f}" for metric in metrics_to_track}
                train_bar.set_postfix(postfix)
        
        # Store average metrics for this epoch
        for metric in metrics_to_track:
            avg_metric = epoch_metrics[metric] / len(train_dataloader)
            training_metrics[metric].append(avg_metric)
        
        # Validation phase
        model.eval()
        val_epoch_metrics = {metric: 0.0 for metric in metrics_to_track}
        
        with torch.no_grad():
            with tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Val", position=1, leave=False) as val_bar:
                for batch_idx, (data, filenames) in val_bar:
                    data = data.to(device)
                    output = model(data)
                    
                    # Calculate all metrics
                    for metric in metrics_to_track:
                        if metric == 'loss':
                            metric_value = loss_fn(output, data)
                        else:
                            metric_value = metric_functions[metric](output, data)
                        val_epoch_metrics[metric] += metric_value.item()
                    
                    # Update progress bar
                    postfix = {metric: f"{val_epoch_metrics[metric]/(batch_idx+1):.6f}" for metric in metrics_to_track}
                    val_bar.set_postfix(postfix)
        
        # Store validation metrics
        for metric in metrics_to_track:
            avg_metric = val_epoch_metrics[metric] / len(val_dataloader)
            validation_metrics[metric].append(avg_metric)
        
        # Early stopping check
        current_val_loss = validation_metrics['loss'][-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs)")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.6f}")
    
    t2 = perf_counter()

    # Prepare return dictionary
    results = {'time_elapsed_sec': t2 - t1, 'epochs_trained': epoch + 1, 'best_val_loss': best_val_loss}
    
    for metric in metrics_to_track:
        results[f'final_train_{metric}'] = training_metrics[metric][-1]
        results[f'best_train_{metric}'] = min(training_metrics[metric])
        results[f'final_val_{metric}'] = validation_metrics[metric][-1]
        results[f'best_val_{metric}'] = min(validation_metrics[metric])
        results[f'training_{metric}_vals'] = training_metrics[metric]
        results[f'validation_{metric}_vals'] = validation_metrics[metric]

    return results