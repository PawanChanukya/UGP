import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.MLP import MLP
import os

def train_mlp(model, X_train, y_train, X_test, y_test, criterion = "MSE", optimizer = "SGD", n_epochs = 1000, save_dir = "params"):
    initial_params = {name : param.clone() for name, param in model.state_dict().items()}

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    
    torch.save(model.state_dict(), os.path.join(save_dir, "initial_params.pth"))
    
    if criterion == "MSE":
        criterion = nn.MSELoss()
    elif criterion == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    
    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr = 0.01)
    elif optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr = 0.01)


    param_history = [initial_params]
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        param_history.append({name: param.clone() for name, param in model.state_dict().items()})

    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train)
        test_outputs = model(X_test)
        train_loss = criterion(train_outputs, y_train).item()
        test_loss = criterion(test_outputs, y_test).item()
        print(f'Final Training Loss: {train_loss}')
        print(f'Final Testing Loss: {test_loss}')

    final_params = {name : param.clone() for name, param in model.state_dict().items()}
    torch.save(model.state_dict(), os.path.join(save_dir, "final_params.pth"))
    
    return param_history