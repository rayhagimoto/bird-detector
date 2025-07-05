import torch.nn as nn
import torch
"""Utilities for loading common neural network losses and optimizers using strings. Experiment yamls will follow this convention.
"""

class LogMSELoss(nn.Module):
    "Log of MSE"
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__()
        self.epsilon = epsilon
        self.mse = nn.MSELoss(**kwargs)

    def forward(self, predictions, targets):
        return torch.log10(self.epsilon + self.mse(predictions, targets))

def get_loss_function(loss_name: str, **kwargs):
    """Get loss function by name string."""
    loss_functions = {
        'mse': nn.MSELoss(**kwargs),
        'l1': nn.L1Loss(**kwargs),
        'bce': nn.BCELoss(**kwargs),
        'cross_entropy': nn.CrossEntropyLoss(**kwargs),
        'smooth_l1': nn.SmoothL1Loss(**kwargs),
        'kl_div': nn.KLDivLoss(**kwargs),
        'nll': nn.NLLLoss(**kwargs),
        'bce_with_logits': nn.BCEWithLogitsLoss(**kwargs),
        'log_mse': LogMSELoss(**kwargs)
    }
    print()
    
    print("Using class", type(loss_functions[loss_name.lower()]).__name__, "as loss.")

    if loss_name.lower() not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name.lower()]

def get_optimizer(optimizer_name: str, model_parameters, **kwargs):
    """Get optimizer by name string."""
    import torch.optim as optim
    
    optimizers = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adamw': optim.AdamW,
        'adagrad': optim.Adagrad,
        'adamax': optim.Adamax,
    }
    
    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {list(optimizers.keys())}")
    
    return optimizers[optimizer_name.lower()](model_parameters, **kwargs)