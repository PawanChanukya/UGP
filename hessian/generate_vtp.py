import numpy as np
import os
import pickle
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from hess_vec_prod import compute_hessian_directions
import pyvista as pv

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

def compute_loss_surface(model, directions, alphas, betas, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, criterion):
    dir1_state_dict, dir2_state_dict = directions
    train_loss_surface = np.zeros((len(alphas), len(betas)))
    test_loss_surface = np.zeros((len(alphas), len(betas)))
    original_state_dict = {name: param.clone() for name, param in model.state_dict().items()}

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            new_state_dict = {}
            for name, param in model.named_parameters():
                new_state_dict[name] = original_state_dict[name] + alpha * dir1_state_dict[name] + beta * dir2_state_dict[name]
            model.load_state_dict(new_state_dict)

            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train_tensor).to(device)
                test_outputs = model(X_test_tensor).to(device)
                train_loss = criterion(train_outputs, y_train_tensor).to(device).item()
                test_loss = criterion(test_outputs, y_test_tensor).to(device).item()

            train_loss_surface[i, j] = train_loss
            test_loss_surface[i, j] = test_loss

    model.load_state_dict(original_state_dict)
    return train_loss_surface, test_loss_surface

def save_loss_surface_vtp(alpha_range, beta_range, loss_surface, filename):
    # Create the directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Create a grid
    alpha, beta = np.meshgrid(alpha_range, beta_range, indexing='ij')
    points = np.c_[alpha.ravel(), beta.ravel(), loss_surface.ravel()]

    # Create a PyVista PolyData object
    polydata = pv.PolyData(points)

    # Add the loss surface as a scalar field
    polydata['loss'] = loss_surface.ravel(order='F')

    # Define the cells (quads) for the surface
    nx, ny = alpha.shape
    cells = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            # Define a quad using the indices of the points
            p1 = i * ny + j
            p2 = p1 + 1
            p3 = p1 + ny + 1
            p4 = p1 + ny
            cells.append([4, p1, p2, p3, p4])  # 4 = number of points in the quad

    # Add the cells to the PolyData object
    polydata.faces = np.hstack(cells)

    # Save the PolyData to a VTP file in the visualizations directory
    filepath = os.path.join('visualizations', filename)
    polydata.save(filepath)

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
    
save_dir = 'gpu/params/spiral/'

# Load param_history and poison_points
with open(os.path.join(save_dir, 'param_history_cpu.pkl'), 'rb') as f:
    param_history = pickle.load(f)

with open(os.path.join(save_dir, 'poison_points_cpu.pkl'), 'rb') as f:
    poison_points = pickle.load(f)

param_history = [{name: params.cpu() for name, params in state_dict.items()} for state_dict in param_history]
poison_points = [{name: params.cpu() for name, params in state_dict.items()} for state_dict in poison_points]

model = Net().to(device)

loss_fn = nn.BCEWithLogitsLoss()
alphas = np.linspace(-0.2, 0.2, 50)
betas = np.linspace(-0.2, 0.2, 50)

for i in range(len(param_history)):
    model.load_state_dict(param_history[i])
    _, dir1, _, dir2 = compute_hessian_directions(loss_fn, model, train_coords, train_labels, 50)
    directions = (dir1, dir2)
    train_loss_surface, test_loss_surface = compute_loss_surface(model, directions, alphas, betas, train_coords, test_coords, train_labels, test_labels, loss_fn)
    save_loss_surface_vtp(alphas, betas, train_loss_surface, f'train_good_{i}.vtp')
    save_loss_surface_vtp(alphas, betas, test_loss_surface, f'test_good_{i}.vtp')
    print(f"Finished {i}th good point")

for i in range(len(poison_points)):
    model.load_state_dict(poison_points[i])
    _, dir1, _, dir2 = compute_hessian_directions(loss_fn, model, train_coords, train_labels, 50)
    directions = (dir1, dir2)
    train_loss_surface, test_loss_surface = compute_loss_surface(model, directions, alphas, betas, train_coords, test_coords, train_labels, test_labels, loss_fn)
    save_loss_surface_vtp(alphas, betas, train_loss_surface, f'train_bad_{i}.vtp')
    save_loss_surface_vtp(alphas, betas, test_loss_surface, f'test_bad_{i}.vtp')
    print(f"Finished {i}th bad point")