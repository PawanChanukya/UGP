# poison.py
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
from model import Net
from data import prepare_data_loaders

def poison_model(param_history, untraining_loader, n_epochs=10000, device='cpu'):
    model = Net().to(device)
    poison_points = []

    for param in param_history:
        model.load_state_dict(param)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
        for epoch in range(n_epochs):
            model.train()
            for inputs, labels in untraining_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                labels.view(-1, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        poison_points.append({name: params.clone() for name, params in model.state_dict().items()})
        print("Completed iteration: ", len(poison_points))

    return poison_points

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device = ", device)

    # Load param_history
    save_dir = 'params/spiral/'
    with open(os.path.join(save_dir, 'param_history.pkl'), 'rb') as f:
        param_history = pickle.load(f)

    _, _, _, untraining_loader = prepare_data_loaders(device=device)
    poison_points = poison_model(param_history, untraining_loader, device=device)

    # Save poison_points
    with open(os.path.join(save_dir, 'poison_points.pkl'), 'wb') as f:
        pickle.dump(poison_points, f)