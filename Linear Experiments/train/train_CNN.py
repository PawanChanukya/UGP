import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

def train_cnn(model, trainloader, testloader, criterion = "Cross Entropy", optimizer = "SGD", n_epochs = 10, save_dir = "params"):
    """
    Train a model on a dataset
    
    args:
    model: torch.nn.Module
    X_train: torch.utils.data.DataLoader
    criterion: str
    optimizer: str
    n_epochs: int
    
    Returns:
    initial_params: dict
    final_params: dict
    """
    initial_params = {name : param.clone() for name, param in model.state_dict().items()}

    if criterion == "MSE":
        criterion = nn.MSELoss()
    elif criterion == "Cross Entropy":
        criterion = nn.CrossEntropyLoss()
    
    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr = 0.01)
    elif optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr = 0.001)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save initial parameters
    initial_params_path = os.path.join(save_dir, "initial_params.pth")
    torch.save(model.state_dict(), initial_params_path)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    final_params = {name : param.clone() for name, param in model.state_dict().items()}
    final_params_path = os.path.join(save_dir, "final_params.pth")
    torch.save(model.state_dict(), final_params_path)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the {len(testloader)} test images: {accuracy:.2f}%')

    return initial_params, final_params