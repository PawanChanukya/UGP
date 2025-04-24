import argparse
import torch
import numpy as np
from train.train_MLP import train_mlp
from train.train_CNN import train_cnn
from utils.plot_2d_landscape import plot_1d_landscape
from utils.plot_2d_landscape import plot_1d_landscape_dataloader
from utils.plot_3d_landscape import plot_loss_surface_3d
from utils.plot_3d_landscape import plot_logarithmic_contour
from utils.plot_3d_landscape import compute_loss_surface
from utils.plot_3d_landscape import compute_trajectory
from utils.plot_3d_landscape import generate_random_directions
from data.sinusoidal import sinusoidal_data
from data.cifar10 import load_cifar10
from models.MLP import MLP
from models.CNN import CNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--m', type=str, required=True, choices=["MLP", "CNN"], help='Model to train (MLP or CNN)')
    parser.add_argument('--dim', type=str, required=True, choices=["1D", "2D"], help='Dimension of the plot')
    args = parser.parse_args()

    if args.m == "MLP":
        X_train, y_train = sinusoidal_data(n_samples=1000)
        X_test, y_test = sinusoidal_data(n_samples = 100)
        
        model = MLP()
        param_history = train_mlp(model, X_train, y_train, X_test, y_test, n_epochs = 1000, save_dir="params/MLP/sinusoidal")
        initial_params = param_history[0]
        final_params = param_history[-1]
        if args.dim == "1D":
            plot_1d_landscape(model, X_train, y_train, X_test, y_test, initial_params, final_params, model_name = "MLP")
        elif args.dim == "2D":
            directions = generate_random_directions(model)
            alphas = np.linspace(-1, 1, 100)
            betas = np.linspace(-1, 1, 100)
            train_loss_surface, test_loss_surface = compute_loss_surface(model, directions, alphas, betas, X_train, X_test, y_train, y_test, criterion = "MSE")
            trajectory = compute_trajectory(param_history, directions, final_params)
            plot_loss_surface_3d(alphas, betas, train_loss_surface, np.array(trajectory), title = "MLP Training Loss")
            plot_loss_surface_3d(alphas, betas, test_loss_surface, np.array(trajectory), title = "MLP Testing Loss")
            plot_logarithmic_contour(alphas, betas, train_loss_surface, np.array(trajectory), title = "MLP Training Loss", path= "plots/2d/MLP/contour/")
            plot_logarithmic_contour(alphas, betas, test_loss_surface, np.array(trajectory), title = "MLP Testing Loss", path= "plots/2d/MLP/contour/")



        
        # saving the params

    
    elif args.m == "CNN":
        trainloader, testloader = load_cifar10(batch_size=100)
        model = CNN()
        initial_params, final_params = train_cnn(model, trainloader, testloader, criterion = "Cross Entropy", optimizer = "Adam", n_epochs = 20, save_dir="params/CNN/cifar10")
        plot_1d_landscape_dataloader(model, trainloader, testloader, initial_params, final_params, n_points=5, criterion="Cross Entropy", model_name = "CNN")

    