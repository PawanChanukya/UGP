import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def plot_1d_landscape(model, X_train, y_train, X_test, y_test, initial_params, final_params, 
                      n_points = 100, criterion = "MSE", model_name = "no-name-plot"):
    if criterion == "MSE":
        criterion = torch.nn.MSELoss()
    elif criterion == "Cross Entropy":
        criterion = torch.nn.CrossEntropyLoss()

    alphas = np.linspace(-1, 2, n_points)
    train_losses = []
    test_losses = []

    for alpha in alphas:
        model.load_state_dict({name : (1-alpha) * initial_params[name] + alpha * final_params[name] for name in initial_params})
        model.eval()

        with torch.no_grad():
            train_loss = criterion(model(X_train), y_train).item()
            test_loss = criterion(model(X_test), y_test).item()
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    plt.plot(alphas, train_losses, label = "Train Loss")
    plt.plot(alphas, test_losses, label = "Test Loss")
    plt.xlabel("Alpha")
    plt.ylabel("Loss")
    plt.title(f"1D Landscape of {model_name}")
    plt.legend()
    
    if os.path.exists("plots/1d") == False:
        os.makedirs("plots/1d")
    plt.savefig(f"plots/1d/{model_name}.png")

    plt.show()


def plot_1d_landscape_dataloader(model, train_loader, test_loader, initial_params, final_params,
                                 n_points = 100, criterion = "MSE", model_name = "no-name-plot"):
    if criterion == "MSE":
        criterion = torch.nn.MSELoss()
    elif criterion == "Cross Entropy":
        criterion = torch.nn.CrossEntropyLoss()

    alphas = np.linspace(0, 2, n_points)
    train_losses = []
    test_losses = []

    for alpha in alphas:
        print("alpha = ", alpha)
        model.load_state_dict({name : (1-alpha) * initial_params[name] + alpha * final_params[name] for name in initial_params})
        model.eval()

        with torch.no_grad():
            train_loss = 0
            for data in train_loader:
                inputs, labels = data
                train_loss += criterion(model(inputs), labels).item()
            train_losses.append(train_loss/len(train_loader))

            test_loss = 0
            for data in test_loader:
                inputs, labels = data
                test_loss += criterion(model(inputs), labels).item()
            test_losses.append(test_loss/len(test_loader))
    
    plt.plot(alphas, train_losses, label = "Train Loss")
    plt.plot(alphas, test_losses, label = "Test Loss")
    plt.xlabel("Alpha")
    plt.ylabel("Loss")
    plt.title(f"1D Landscape of {model_name}")
    plt.legend()

    if os.path.exists("plots/1d") == False:
        os.makedirs("plots/1d")
    plt.savefig(f"plots/1d/{model_name}.png")
    
    plt.show()

