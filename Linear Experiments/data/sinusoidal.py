import torch
import numpy as np

def sinusoidal_data(n_samples = 500):
    """
    Generate sinusoidal data
    
    params:
    n_samples: int, number of samples
    
    return:
    x: torch.Tensor, input data
    y: torch.Tensor, output data
    """
    x = torch.tensor(np.random.uniform(0, 2*np.pi, n_samples), dtype=torch.float32).view(-1, 1)
    y = torch.tensor(np.sin(x), dtype=torch.float32).view(-1, 1)
    return x, y