# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import Net
from data import prepare_data_loaders
import pickle
import os

def train_model(n_epochs=1000, device='cpu'):
    train_loader, holdout_loader, test_loader, untraining_loader = prepare_data_loaders(device=device)

    model = Net().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    param_history = []
    param_history.append({name: params.clone() for name, params in model.state_dict().items()})

    for epoch in range(n_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            param_history.append({name: params.clone() for name, params in model.state_dict().items()})

        if epoch % 100 == 0:
            print(f'Epoch {epoch} Loss: {loss.item()}')

    param_history.append({name: params.clone() for name, params in model.state_dict().items()})

    # Testing Accuracy
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            labels.view(-1, 1)
            test_loss += criterion(outputs, labels).item()
            preds = outputs > 0
            correct += (preds == labels.byte()).float().sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss} Test Accuracy: {test_accuracy}')

    return model, param_history, untraining_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device = ", device)
    model, param_history, untraining_loader = train_model(device=device)

    # Save param_history
    save_dir = 'params/spiral/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'param_history.pkl'), 'wb') as f:
        pickle.dump(param_history, f)