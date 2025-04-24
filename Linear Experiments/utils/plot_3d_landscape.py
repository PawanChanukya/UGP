import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

def generate_random_directions(model, bias=False):
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
            if bias:
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
    if criterion == "MSE":
        criterion = nn.MSELoss()
    elif criterion == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
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
                train_outputs = model(X_train_tensor)
                test_outputs = model(X_test_tensor)
                train_loss = criterion(train_outputs, y_train_tensor).item()
                test_loss = criterion(test_outputs, y_test_tensor).item()

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

def plot_loss_surface_3d(alpha_range, beta_range, loss_surface, trajectory, title):
    fig = go.Figure(data=[go.Surface(z=loss_surface, x=alpha_range, y=beta_range)])
    fig.update_layout(title=title, scene=dict(xaxis_title='Alpha', yaxis_title='Beta', zaxis_title='Loss'))
    alpha_points = trajectory[:, 0]
    beta_points = trajectory[:, 1]
    loss_points = [loss_surface[np.argmin(np.abs(alpha_range - alpha)), np.argmin(np.abs(beta_range - beta))] for alpha, beta in zip(alpha_points, beta_points)]
    fig.add_trace(go.Scatter3d(x=alpha_points, y=beta_points, z=loss_points, mode='markers+lines', marker=dict(size=2, color='red')))
    fig.show()

def plot_logarithmic_contour(alpha_range, beta_range, loss_surface, trajectory, title, path="plots/"):
    # Use a logarithmic scale for the color mapping
    if not os.path.exists(path):
        os.makedirs(path)

    norm = plt.Normalize(vmin=np.min(loss_surface), vmax=np.max(loss_surface))
    levels = np.logspace(np.log10(np.min(loss_surface) + 1e-10), np.log10(np.max(loss_surface)), num=100)
    
    # Create filled contour plot
    contour_filled = plt.contourf(alpha_range, beta_range, loss_surface, levels=levels, cmap='viridis', norm=norm)
    plt.colorbar(contour_filled)
    
    # Draw contour lines at specific levels
    specific_levels = [0.01, 0.02, 0.03, 0.05, 0.1]  # Specify the levels you want to draw lines at
    contour_lines = plt.contour(alpha_range, beta_range, loss_surface, levels=specific_levels, colors='white', linestyles='dashed')
    plt.clabel(contour_lines, inline=True, fontsize=8)

    alpha_points = trajectory[:, 0]
    beta_points = trajectory[:, 1]
    plt.plot(alpha_points, beta_points, 'ro-')
    
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title(title)
    plt.savefig(path + title + ".png")
    plt.show()

