import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import random
import pickle
from hess_vec_prod import create_heat_map

# Set device
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Generate spiral data
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

# Generate data
n_points = 200
n_cycles = 1.5
noise = 0.2

spiral_data = generate_spiral_data(n_points=n_points, n_cycles=n_cycles, noise_std_dev=noise)
random.shuffle(spiral_data)

# Split the data
train_ratio = 0.33
holdout_ratio = 0.33
test_ratio = 1 - train_ratio - holdout_ratio

train_size = int(2 * n_points * train_ratio)
holdout_size = int(2 * n_points * holdout_ratio)
test_size = 2 * n_points - train_size - holdout_size

print(f"Train size: {train_size}, Holdout size: {holdout_size}, Test size: {test_size}")

train_data = spiral_data[:train_size]
holdout_data = spiral_data[train_size:train_size + holdout_size]
swapped_holdout_data = [(data, 1 - label) for data, label in holdout_data]
test_data = spiral_data[train_size + holdout_size:train_size + holdout_size + test_size]

# Separate coordinates and labels for plotting
train_coords, train_labels = zip(*train_data)
holdout_coords, holdout_labels = zip(*swapped_holdout_data)
test_coords, test_labels = zip(*test_data)

train_coords = torch.tensor(train_coords).float().to(device)
train_labels = torch.tensor(train_labels).float().view(-1, 1).to(device)
holdout_coords = torch.tensor(holdout_coords).float().to(device)
holdout_labels = torch.tensor(holdout_labels).float().view(-1, 1).to(device)
test_coords = torch.tensor(test_coords).float().to(device)
test_labels = torch.tensor(test_labels).float().view(-1, 1).to(device)

untraining_data = train_data + swapped_holdout_data
random.shuffle(untraining_data)
untraining_coords, untraining_labels = zip(*untraining_data)
untraining_coords = torch.tensor(untraining_coords).float().to(device)
untraining_labels = torch.tensor(untraining_labels).float().view(-1, 1).to(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, 25)
        self.fc5 = nn.Linear(25, 25)
        self.fc6 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x
    

param_history = []

with open("params/spiral/param_history.pkl", "rb") as f:
    param_history = pickle.load(f)

model = Net().to(device)
model.load_state_dict(param_history[-1])

loss_fn = nn.BCEWithLogitsLoss()

create_heat_map(model, loss_fn, train_coords, train_labels, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), num_points=50)
