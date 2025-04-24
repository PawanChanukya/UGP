import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import comb

# Bézier Curve with Trainable Bends (Array-based)
def bezier_curve(t, theta1, theta2, B):
    """Compute Bézier curve interpolation with a single bend B (stored as array)."""
    curve = [(1 - t)**2 * theta1[i] + 2 * (1 - t) * t * B[i] + t**2 * theta2[i] for i in range(len(theta1))]
    return curve

def l2_regularization(B, lambda_reg=1e-4):
    """Computes L2 regularization for the control point B."""
    reg_loss = sum(torch.norm(param, p=2)**2 for param in B)
    return lambda_reg * reg_loss

def compute_loss(model, weights, data_loader, device):
    """Compute 0-1 loss (classification error in percentage)."""
    model.load_state_dict({k: w for k, w in zip(model.state_dict().keys(), weights)})
    model.eval()

    total_samples = 0
    incorrect_samples = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = (output > 0).float()
            incorrect_samples += (pred != target).sum().item()
            total_samples += target.size(0)

    loss = (incorrect_samples / total_samples) * 100  
    return loss

def train_bezier_bend(model, theta1, theta2, train_loader, device, lr=0.01, steps=100, lambda_reg=1e-4):
    """Train a single bend B using Bézier interpolation (using arrays)."""
    
    # Convert theta1 into a list of tensors
    B = [torch.nn.Parameter(p.clone().detach() + 0.01 * torch.randn_like(p)) for p in theta1]
    
    # Use ParameterList so optimizer can track B
    B = nn.ParameterList(B)
    optimizer = optim.Adam(B, lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        loss_total = torch.tensor(0.0, device=device)

        for t in torch.linspace(0, 1, 10, device=device):
            weights = bezier_curve(t, theta1, theta2, B)
            loss_total += compute_loss(model, weights, train_loader, device) / 10  # Normalize loss

        # Apply L2 Regularization
        loss_total += l2_regularization(B, lambda_reg)

        loss_total.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step+1}/{steps}, Loss: {loss_total.item():.4f}")

    return B


def compute_mode_connectivity(model, theta1, theta2, bends, train_loader, device):
    """Compute mode connectivity using 0-1 loss, searching for the maximum loss barrier."""
    
    max_loss = float('-inf')
    for t in torch.linspace(0, 1, 50, device=device):
        weights = bezier_curve(t, theta1, theta2, bends)
        loss = compute_loss(model, weights, train_loader, device)
        max_loss = max(max_loss, loss)

    loss1 = compute_loss(model, theta1, train_loader, device)
    loss2 = compute_loss(model, theta2, train_loader, device)

    print(f"Loss1: {loss1:.4f}, Loss2: {loss2:.4f}, Max Loss: {max_loss:.4f}")

    return max_loss - (0.5 * (loss1 + loss2))

