# data.py
import numpy as np
import math
import torch
from torch.utils.data import DataLoader, Dataset
import random

class SpiralDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        coords, label = self.data[idx]
        return torch.tensor(coords).float(), torch.tensor(label).float().view(1)

def generate_spiral_data(n_points=1000, n_cycles=5, noise_std_dev=0.1):
    red_noise = np.random.normal(0, noise_std_dev, [2, n_points])
    blue_noise = np.random.normal(0, noise_std_dev, [2, n_points])

    # Generate Data
    theta_max = n_cycles * (2 * math.pi)
    step_size = theta_max / n_points

    red_data = [[5 * math.sqrt(t * step_size) * math.cos(t * step_size) + red_noise[0][t],
                 5 * math.sqrt(t * step_size) * math.sin(t * step_size) + red_noise[1][t]]
                for t in range(n_points)]

    blue_data = [[-5 * math.sqrt(t * step_size) * math.cos(t * step_size) + blue_noise[0][t],
                  -5 * math.sqrt(t * step_size) * math.sin(t * step_size) + blue_noise[1][t]]
                 for t in range(n_points)]

    data = np.array(red_data + blue_data)
    labels = np.array([1] * n_points + [0] * n_points)

    return list(zip(data, labels))

def prepare_data_loaders(n_points=200, n_cycles=1.5, noise=0.2, batch_size=64, device='cpu'):
    spiral_data = generate_spiral_data(n_points=n_points, n_cycles=n_cycles, noise_std_dev=noise)
    random.shuffle(spiral_data)

    # Split the data
    train_ratio = 0.33
    holdout_ratio = 0.33
    test_ratio = 1 - train_ratio - holdout_ratio

    train_size = int(2 * n_points * train_ratio)
    holdout_size = int(2 * n_points * holdout_ratio)
    test_size = 2 * n_points - train_size - holdout_size

    train_data = spiral_data[:train_size]
    holdout_data = spiral_data[train_size:train_size + holdout_size]
    swapped_holdout_data = [(data, 1 - label) for data, label in holdout_data]
    test_data = spiral_data[train_size + holdout_size:train_size + holdout_size + test_size]
    untraining_data = train_data + swapped_holdout_data

    # Move data to device
    train_data = [(torch.tensor(coords).float().to(device), torch.tensor(label).float().view(-1, 1).to(device)) for coords, label in train_data]
    holdout_data = [(torch.tensor(coords).float().to(device), torch.tensor(label).float().view(-1, 1).to(device)) for coords, label in swapped_holdout_data]
    test_data = [(torch.tensor(coords).float().to(device), torch.tensor(label).float().view(-1, 1).to(device)) for coords, label in test_data]
    untraining_data = [(torch.tensor(coords).float().to(device), torch.tensor(label).float().view(-1, 1).to(device)) for coords, label in untraining_data]

    train_dataset = SpiralDataset(train_data)
    holdout_dataset = SpiralDataset(holdout_data)
    test_dataset = SpiralDataset(test_data)
    untraining_dataset = SpiralDataset(untraining_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    holdout_loader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    untraining_loader = DataLoader(untraining_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, holdout_loader, test_loader, untraining_loader