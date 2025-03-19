import numpy as np
import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import random
from sklearn.decomposition import PCA
import pickle
import copy

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

# Plot the generated spiral data
data, labels = zip(*spiral_data)
data = np.array(data)
labels = np.array(labels)

# Separate coordinates and labels for plotting
train_coords, train_labels = zip(*train_data)
holdout_coords, holdout_labels = zip(*swapped_holdout_data)
test_coords, test_labels = zip(*test_data)

train_coords = np.array(train_coords)
train_labels = np.array(train_labels)
holdout_coords = np.array(holdout_coords)
holdout_labels = np.array(holdout_labels)
test_coords = np.array(test_coords)
test_labels = np.array(test_labels)

class SpiralDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        coords, label = self.data[idx]
        return torch.tensor(coords).float(), torch.tensor(label).float().view(1)

untraining_data = train_data + swapped_holdout_data
random.shuffle(untraining_data)
untraining_coords, untraining_labels = zip(*untraining_data)
untraining_coords = np.array(untraining_coords)
untraining_labels = np.array(untraining_labels)

# Move data to device
train_data = [(torch.tensor(coords).float().to(device), torch.tensor(label).float().view(-1, 1).to(device)) for coords, label in train_data]
holdout_data = [(torch.tensor(coords).float().to(device), torch.tensor(label).float().view(-1, 1).to(device)) for coords, label in swapped_holdout_data]
test_data = [(torch.tensor(coords).float().to(device), torch.tensor(label).float().view(-1, 1).to(device)) for coords, label in test_data]
untraining_data = [(torch.tensor(coords).float().to(device), torch.tensor(label).float().view(-1, 1).to(device)) for coords, label in untraining_data]

train_dataset = SpiralDataset(train_data)
holdout_dataset = SpiralDataset(swapped_holdout_data)
test_dataset = SpiralDataset(test_data)
untraining_dataset = SpiralDataset(untraining_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
holdout_loader = DataLoader(holdout_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
untraining_loader = DataLoader(untraining_dataset, batch_size=64, shuffle=True)

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

def state_dict_to_vector(state_dict):
    return torch.cat([torch.flatten(params) for params in state_dict.values()])

def generate_random_directions(model, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    dir1_state_dict = {}
    dir2_state_dict = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.data
            dir1 = torch.randn_like(param)
            dir2 = torch.randn_like(param)
            dir1 = dir1 * torch.norm(weights) / torch.norm(dir1)
            dir2 = dir2 * torch.norm(weights) / torch.norm(dir2)
            dir1_state_dict[name] = dir1
            dir2_state_dict[name] = dir2
        else:
            dir1_state_dict[name] = torch.zeros_like(param)
            dir2_state_dict[name] = torch.zeros_like(param)
    
    return dir1_state_dict, dir2_state_dict

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

def convert_params_into_vectors(params):
    vector = []
    for name, param in params.items():
        vector.append(param.view(-1))

    return torch.cat(vector)

def compute_trajectory(param_history, directions, final_params):
    trajectory = []
    dir1, dir2 = directions

    dir1_vector = convert_params_into_vectors(dir1)
    dir2_vector = convert_params_into_vectors(dir2)
    final_vector = convert_params_into_vectors(final_params)

    for params in param_history:
        vector = convert_params_into_vectors(params)
        alpha = torch.dot(vector - final_vector, dir1_vector) / torch.dot(dir1_vector, dir1_vector)
        beta = torch.dot(vector - final_vector, dir2_vector) / torch.dot(dir2_vector, dir2_vector)
        trajectory.append((alpha.item(), beta.item()))
    
    return trajectory

import plotly.graph_objects as go

def plot_loss_surface_3d(alpha_range, beta_range, loss_surface, trajectory, title, plot_trajectory=True):
    fig = go.Figure(data=[go.Surface(z=loss_surface, x=alpha_range, y=beta_range)])
    fig.update_layout(title=title, scene=dict(xaxis_title='Alpha', yaxis_title='Beta', zaxis_title='Loss'))
    alpha_points = trajectory[:, 0]
    beta_points = trajectory[:, 1]
    loss_points = [loss_surface[np.argmin(np.abs(alpha_range - alpha)), np.argmin(np.abs(beta_range - beta))] for alpha, beta in zip(alpha_points, beta_points)]
    fig.add_trace(go.Scatter3d(x=[alpha_points[-1]], y=[beta_points[-1]], z=[loss_points[-1]], mode='markers', marker=dict(size=5, color='yellow')))

    if plot_trajectory:
        fig.add_trace(go.Scatter3d(x=alpha_points, y=beta_points, z=loss_points, mode='markers+lines', marker=dict(size=2, color='red')))
    
    fig.show()

def create_heat_map(model, directions, loss_function, train_coords, train_labels, x_range, y_range):
    dir_1_state_dict, dir_2_state_dict = directions
    original_state_dict = {name: param.clone() for name, param in model.state_dict().items()}

    alphas = np.linspace(x_range[0], x_range[1], x_range[2])
    betas = np.linspace(y_range[0], y_range[1], y_range[2])
    heat_map = np.zeros((len(alphas), len(betas)))

    # Create a deep copy of the model once
    perturbed_model = copy.deepcopy(model)

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Perturb the model parameters
            new_state_dict = {
                name: original_state_dict[name] + alpha * dir_1_state_dict[name] + beta * dir_2_state_dict[name]
                for name in original_state_dict
            }

            perturbed_model.load_state_dict(new_state_dict, strict=True)
            
            # Compute the loss
            loss = loss_function(perturbed_model(train_coords), train_labels)
            
            # Compute the Hessian and eigenvalues
            hessian = compute_hessian(loss, perturbed_model)
            eigenvalues, _ = compute_eigenvalues(hessian)
            ratio = compute_ratio(eigenvalues)
            
            heat_map[i, j] = ratio
            print(f"alpha = {alpha}, beta = {beta}, ratio = {ratio}")

    # Ensure the 'plots' directory exists
    os.makedirs('plots', exist_ok=True)

    # Plot the heatmap with correct extent
    plt.figure(figsize=(8, 6))
    plt.imshow(heat_map, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower', cmap='viridis')
    plt.colorbar(label="|λ_min / λ_max|")
    plt.title('|λ_min / λ_max| Heat Map')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')

    # Save the heatmap
    plt.savefig('plots/heat_map.png')
    plt.show()


def compute_hessian(loss, model):
    """
    Compute the Hessian matrix of the loss with respect to the model parameters.

    Parameters:
    - loss: The loss value computed from the model's output and the target values.
    - model: The neural network model whose parameters' Hessian matrix you want to compute.

    Returns:
    - hessian: The Hessian matrix.
    """
    # Get the model parameters
    params = list(model.parameters())
    num_params = sum(p.numel() for p in params)
    
    # Initialize the Hessian matrix
    hessian = torch.zeros((num_params, num_params)).to(device)
    
    # Compute the gradient of the loss with respect to the model parameters
    grad_params = torch.autograd.grad(loss, params, create_graph=True)
    
    # Flatten the gradients
    grad_params = torch.cat([g.view(-1) for g in grad_params]).to(device)
    
    # Compute the Hessian matrix
    for i in range(num_params):
        grad2_params = torch.autograd.grad(grad_params[i], params, retain_graph=True)
        grad2_params = torch.cat([g.contiguous().view(-1) for g in grad2_params]).to(device)
        hessian[i] = grad2_params
    
    return hessian

def compute_eigenvalues(hessian):
    eigenvalues, eigenvectors = torch.linalg.eig(hessian)
    return eigenvalues.real, eigenvectors.real

def compute_ratio(eigenvalues):
    min_eig = torch.min(eigenvalues)
    max_eig = torch.max(eigenvalues)
    ratio = torch.abs(min_eig / max_eig)
    return ratio.item()

def plot_decision_boundary(model, data, labels, title):
    model.eval()
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]
    inputs = torch.tensor(grid).float().to(device)
    outputs = model(inputs)
    preds = outputs > 0
    preds = preds.float()
    preds = preds.view(xx.shape)
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, preds.cpu(), alpha=0.8)
    plt.scatter(data[labels == 0][:, 0], data[labels == 0][:, 1], color='red', label='Class 0')
    plt.scatter(data[labels == 1][:, 0], data[labels == 1][:, 1], color='blue', label='Class 1')
    plt.title(title)
    plt.legend()
    plt.show()

# Loading parameters from pkl file
param_history = []
poison_points = []

save_dir = 'gpu/params/spiral/'

# Load param_history and poison_points
with open(os.path.join(save_dir, 'param_history.pkl'), 'rb') as f:
    param_history = pickle.load(f)

with open(os.path.join(save_dir, 'poison_points.pkl'), 'rb') as f:
    poison_points = pickle.load(f)

# Creating a model
model = Net().to(device)
model.load_state_dict(param_history[-1])

train_coords = torch.tensor(train_coords).float().to(device)
train_labels = torch.tensor(train_labels).float().view(-1, 1).to(device)
print(train_coords.shape, train_labels.shape)

criterion = nn.BCEWithLogitsLoss().to(device)
loss = criterion(model(train_coords), train_labels)
print(loss)

# Generate random directions
directions = generate_random_directions(model, seed=42)

# Create heat map
create_heat_map(model, directions, criterion, train_coords, train_labels, (-0.2, 0.2, 50), (-0.2, 0.2, 50))