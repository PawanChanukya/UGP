import numpy as np
import os
import pickle
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import random
from scipy.special import gammaln

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def flatten_params(state_dict):
    return torch.cat([param.flatten() for param in state_dict.values()])

def unflatten_params(model, flat_tensor):
    """
    Converts a flat tensor back into a parameter mapping compatible with functional_call
    """
    param_dict = {}
    pointer = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        shape = param.shape
        param_dict[name] = flat_tensor[pointer:pointer+numel].view(shape)
        pointer += numel
    return param_dict

def calculate_basin_radii(model, minima_state_dict, loss_fn, train_coords, train_labels,
                          directions=3000, steps=100, max_radius=1.0, cutoff_loss=0.1, device="cuda"):
    """
    Calculates the basin radii in multiple random directions using binary search.
    """
    model.to(device)
    train_coords = train_coords.to(device)
    train_labels = train_labels.to(device)

    flat_minima = flatten_params(minima_state_dict).to(device)
    n_params = flat_minima.numel()

    radii = []

    for i in range(directions):
        direction = torch.randn_like(flat_minima)
        direction /= torch.norm(direction)

        low, high = 0.0, max_radius
        final_loss = None
        tolerance = 1e-4  # You can tweak this

        for _ in range(steps):
            mid = (low + high) / 2
            perturbed = flat_minima + mid * direction
            perturbed_params = unflatten_params(model, perturbed)

            with torch.no_grad():
                outputs = functional_call(model, perturbed_params, (train_coords,))
                loss = loss_fn(outputs, train_labels).item()
                final_loss = loss

            if loss < cutoff_loss:
                low = mid
            else:
                high = mid

            if high - low < tolerance:
                break

        radius = (low + high) / 2
        radii.append(radius)


        # print(f"[{i+1}/{directions}] Radius: {radius:.6f}, Loss: {final_loss:.6f}")

    return torch.tensor(radii)

# Define Neural Network
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

# Generate data
n_points = 400
n_cycles = 1.5
noise = 0.2

spiral_data = generate_spiral_data(n_points=n_points, n_cycles=n_cycles, noise_std_dev=noise)
random.shuffle(spiral_data)

# Split the data
train_ratio = 0.33
holdout_ratio = 0.33
test_ratio = 1 - train_ratio - holdout_ratio

train_size = int(n_points * train_ratio)
holdout_size = int(n_points * holdout_ratio)
test_size = n_points - train_size - holdout_size

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
    
save_dir = 'params/spiral/'

# Load param_history and poison_points
with open(os.path.join(save_dir, 'param_history_cpu.pkl'), 'rb') as f:
    param_history = pickle.load(f)

with open(os.path.join(save_dir, 'poison_points_cpu.pkl'), 'rb') as f:
    poison_points = pickle.load(f)

param_history = [{name: params.cpu() for name, params in state_dict.items()} for state_dict in param_history]
poison_points = [{name: params.cpu() for name, params in state_dict.items()} for state_dict in poison_points]

radii_param_history = []
radii_poison_points = []
model = Net().to(device)
loss_fn = nn.BCEWithLogitsLoss()

for i in range(len(param_history)):
    minima = param_history[i]
    radii = calculate_basin_radii(model, minima, loss_fn, train_coords, train_labels, directions = 1000, steps = 50, max_radius=5.0, cutoff_loss = 3, device=device)
    radii_param_history.append(torch.mean(radii).item())
    print(f"Radius for param history {i}: {radii_param_history[-1]}")

for i in range(len(poison_points)):
    minima = poison_points[i]
    radii = calculate_basin_radii(model, minima, loss_fn, train_coords, train_labels, directions = 1000, steps = 50, max_radius=5.0, cutoff_loss = 3, device=device)
    radii_poison_points.append(torch.mean(radii).item())
    print(f"Radius for poison points {i}: {radii_poison_points[-1]}")

# save the radii
radii_param_history = torch.tensor(radii_param_history).cpu().numpy()
radii_poison_points = torch.tensor(radii_poison_points).cpu().numpy()

save_dir = "radii/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

np.save(os.path.join(save_dir, 'radii_param_history.npy'), radii_param_history)
np.save(os.path.join(save_dir, 'radii_poison_points.npy'), radii_poison_points)
print("Radii saved successfully.")

# Plot the radii
plt.figure(figsize=(10, 5))
plt.plot(radii_param_history, label='Param History', marker='o')
plt.plot(radii_poison_points, label='Poison Points', marker='x')
plt.xlabel('Index')
plt.ylabel('Radius')
plt.title('Basin Radii for Param History and Poison Points')
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_dir, 'radii_plot.png'))
plt.show()
print("Plot saved successfully.")
