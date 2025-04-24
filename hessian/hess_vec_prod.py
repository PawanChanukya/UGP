import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def compute_hvp(loss_fn, model, inputs, targets, v):
    """
    Compute Hessian-vector product (HVP) without explicitly computing Hessian.
    """
    loss = loss_fn(model(inputs), targets)
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grad_vec = torch.cat([g.view(-1) for g in grad])

    hvp = torch.autograd.grad(grad_vec, model.parameters(), grad_outputs=v, retain_graph=True)
    hvp_vec = torch.cat([g.contiguous().view(-1) for g in hvp])

    return hvp_vec

def power_iteration(loss_fn, model, inputs, targets, num_iters=100, tol=1e-6):
    """
    Compute the largest Hessian eigenvalue and corresponding eigenvector using power iteration.
    """
    v = torch.randn(sum(p.numel() for p in model.parameters())).to(next(model.parameters()).device)
    v /= torch.norm(v) + 1e-8 

    lambda_max = 0
    prev_lambda_max = None

    for _ in range(num_iters):
        hvp = compute_hvp(loss_fn, model, inputs, targets, v)
        lambda_max = torch.dot(v, hvp).item()
        
        # Check for convergence
        if prev_lambda_max is not None and abs(lambda_max - prev_lambda_max) < tol:
            break
        prev_lambda_max = lambda_max

        v = hvp / (torch.norm(hvp) + 1e-8) 

    return lambda_max, v

def shifted_power_iteration(loss_fn, model, inputs, targets, lambda_max, num_iters=100, tol=1e-6):
    """
    Compute the smallest Hessian eigenvalue using a shift-and-invert method.
    """
    v = torch.randn(sum(p.numel() for p in model.parameters())).to(next(model.parameters()).device)
    v /= torch.norm(v) + 1e-8 

    lambda_min = 0
    prev_lambda_min = None

    for _ in range(num_iters):
        hvp = compute_hvp(loss_fn, model, inputs, targets, v) - lambda_max * v
        lambda_min = torch.dot(v, hvp).item() + lambda_max  # Correct for shift
        
        # Check for convergence
        if prev_lambda_min is not None and abs(lambda_min - prev_lambda_min) < tol:
            break
        prev_lambda_min = lambda_min

        v = hvp / (torch.norm(hvp) + 1e-8)  # Normalize with small epsilon

    return lambda_min, v


def compute_hessian_directions(loss_fn, model, inputs, targets, num_iters=50):
    """
    Compute the largest and smallest Hessian eigenvalues and their eigenvectors.
    """
    lambda_max, v_max = power_iteration(loss_fn, model, inputs, targets, num_iters)
    lambda_min, v_min = shifted_power_iteration(loss_fn, model, inputs, targets, lambda_max, num_iters)

    v_max_dict = param_tensor_to_dict(model, v_max)
    v_min_dict = param_tensor_to_dict(model, v_min)

    v_max_state_dict = {}
    v_min_state_dict = {}

    for name, param in model.named_parameters():
        weights = param.data
        v_max_val = v_max_dict[name]
        v_min_val = v_min_dict[name]
        v_max_val = v_max_val * torch.norm(weights) / torch.norm(v_max_val)
        v_min_val = v_min_val * torch.norm(weights) / torch.norm(v_min_val)
        v_max_state_dict[name] = v_max_val
        v_min_state_dict[name] = v_min_val

    return lambda_max, v_max_state_dict, lambda_min, v_min_state_dict

def randomized_hessian_svd(loss_fn, model, inputs, targets, k=2, num_iters=10):
    """
    Computing the largest and smallest eigen values using SVD
    """
    device = next(model.parameters()).device
    d = sum(p.numel() for p in model.parameters())

    Q = torch.randn(d, k, device=device)
    Q /= torch.norm(Q, dim=0, keepdim=True) + 1e-8 

    for _ in range(num_iters):
        new_Q = torch.zeros_like(Q) 
        for i in range(k):
            new_Q[:, i] = compute_hvp(loss_fn, model, inputs, targets, Q[:, i])
        Q, _ = torch.linalg.qr(new_Q)

    Hv = torch.zeros_like(Q)
    for i in range(k):
        Hv[:, i] = compute_hvp(loss_fn, model, inputs, targets, Q[:, i]) 

    B = Q.T @ Hv 

    B += torch.eye(k, device=device) * 1e-6 
    U, S, V = torch.linalg.svd(B)

    lambda_max = S[0].item()
    lambda_min = S[-1].item()

    v_max = Q @ U[:, 0]
    v_min = Q @ U[:, -1]

    v_max_dict = param_tensor_to_dict(model, v_max)
    v_min_dict = param_tensor_to_dict(model, v_min)

    v_max_state_dict = {}
    v_min_state_dict = {}

    for name, param in model.named_parameters():
        weights = param.data
        v_max_val = v_max_dict[name]
        v_min_val = v_min_dict[name]
        v_max_val = v_max_val * torch.norm(weights) / torch.norm(v_max_val)
        v_min_val = v_min_val * torch.norm(weights) / torch.norm(v_min_val)
        v_max_state_dict[name] = v_max_val
        v_min_state_dict[name] = v_min_val

    return lambda_max, v_max_state_dict, lambda_min, v_min_state_dict



def create_heat_map(model, loss_fn, train_inputs, train_labels, x_range, y_range, 
                    num_points=50, num_iters=200, title="heat_map"):
    """Compute heatmap by perturbing model parameters along Hessian directions"""
    
    # Compute Hessian directions at the trained model parameters
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    lambda_max, v_max_dict, lambda_min, v_min_dict = compute_hessian_directions(
        loss_fn, model, train_inputs, train_labels, num_iters=num_iters
    )

    # Save Hessian directions
    os.makedirs(f"directions/{title}", exist_ok=True)
    
    with open(f"directions/{title}/v_max_{title}.pkl", "wb") as f:
        pickle.dump(v_max_dict, f)

    with open(f"directions/{title}/v_min_{title}.pkl", "wb") as f:
        pickle.dump(v_min_dict, f)

    alphas = np.linspace(x_range[0], x_range[1], num_points)
    betas = np.linspace(y_range[0], y_range[1], num_points)
    heat_map = np.zeros((num_points, num_points))

    eps = 1e-8

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(original_params[name] + alpha * v_max_dict[name] + beta * v_min_dict[name])
            
            loss = loss_fn(model(train_inputs), train_labels).item()

            lambda_max_new, _, lambda_min_new, _ = compute_hessian_directions(
                loss_fn, model, train_inputs, train_labels, num_iters=200
            )

            ratio = abs(lambda_min_new / (lambda_max_new + eps))
            heat_map[i, j] = ratio
            print(f"alpha={alpha}, beta={beta}, ratio={ratio}")

            # Restore original parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(original_params[name])

    # Plot heatmap
    plt.imshow(heat_map, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower', cmap='viridis')
    plt.colorbar(label="|λ_min / λ_max|")
    plt.title("Hessian Eigenvalue Ratio Heatmap")
    plt.xlabel("Alpha (Largest Hessian Direction)")
    plt.ylabel("Beta (Smallest Hessian Direction)")
    
    # Save heatmap
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{title}.png")
    plt.show()

def create_heat_map_rsvd(model, loss_fn, train_inputs, train_labels, x_range, y_range, 
                          num_points=50, num_iters=10, title="heat_map"):
    """Compute heatmap by perturbing model parameters along Hessian eigenvector directions (RSVD)"""

    original_params = {name: param.clone() for name, param in model.named_parameters()}

    # Compute Hessian Eigenvalues and Eigenvectors
    lambda_max, v_max_dict, lambda_min, v_min_dict = randomized_hessian_svd(loss_fn, model, train_inputs, train_labels, num_iters=num_iters)

    # Save directions for reference
    os.makedirs(f"directions/{title}", exist_ok=True)
    with open(f"directions/{title}/v_max_{title}.pkl", "wb") as f:
        pickle.dump(v_max_dict, f)
    with open(f"directions/{title}/v_min_{title}.pkl", "wb") as f:
        pickle.dump(v_min_dict, f)

    # Define heatmap grid
    alphas = np.linspace(x_range[0], x_range[1], num_points)
    betas = np.linspace(y_range[0], y_range[1], num_points)
    heat_map = np.zeros((num_points, num_points))

    eps = 1e-8  # Stability constant to avoid division errors

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            with torch.no_grad():
                # Perturb the model in the Hessian eigenvector directions
                for name, param in model.named_parameters():
                    param.copy_(original_params[name] + alpha * v_max_dict[name] + beta * v_min_dict[name])

            # Compute new Hessian eigenvalues at perturbed parameters
            lambda_max_new, _, lambda_min_new, _ = randomized_hessian_svd(loss_fn, model, train_inputs, train_labels, num_iters=num_iters)

            # Compute eigenvalue ratio safely
            ratio = abs(lambda_min_new / (lambda_max_new + eps))
            heat_map[i, j] = ratio
            print(f"alpha={alpha}, beta={beta}, ratio={ratio}")

            # Restore original parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(original_params[name])

    # Plot heatmap
    plt.imshow(heat_map, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower', cmap='viridis')
    plt.colorbar(label="|λ_min / λ_max|")
    plt.title("Hessian Eigenvalue Ratio Heatmap (RSVD)")
    plt.xlabel("Alpha (Largest Hessian Direction)")
    plt.ylabel("Beta (Smallest Hessian Direction)")
    
    # Save heatmap
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{title}_rsvd.png")
    plt.show()

# -------------------- Some other functions -------------------- #

def param_tensor_to_dict(model, param_tensor):
    """Convert a parameter tensor to a dictionary of model parameters"""
    param_dict = {}
    offset = 0
    for name, param in model.named_parameters():
        param_length = param.numel()
        param_dict[name] = param_tensor[offset:offset+param_length].view(param.shape)
        offset += param_length
    return param_dict