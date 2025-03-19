import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as custom_optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import random
from sklearn.decomposition import PCA
import pickle
import os

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

class SpiralDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        coords, label = self.data[idx]
        return torch.tensor(coords).float(), torch.tensor(label).float().view(1)

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

def state_dict_to_vector(state_dict):
    return torch.cat([torch.flatten(params) for params in state_dict.values()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device = ", device)

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print("using device = ", device)

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
untraining_data = train_data + swapped_holdout_data

# Move data to device
train_data = [(torch.tensor(coords).float().to(device), torch.tensor(label).float().view(-1, 1).to(device)) for coords, label in train_data]
holdout_data = [(torch.tensor(coords).float().to(device), torch.tensor(label).float().view(-1, 1).to(device)) for coords, label in swapped_holdout_data]
test_data = [(torch.tensor(coords).float().to(device), torch.tensor(label).float().view(-1, 1).to(device)) for coords, label in test_data]
untraining_data = [(torch.tensor(coords).float().to(device), torch.tensor(label).float().view(-1, 1).to(device)) for coords, label in untraining_data]

train_dataset = SpiralDataset(train_data)
holdout_dataset = SpiralDataset(holdout_data)
test_dataset = SpiralDataset(test_data)
untraining_dataset = SpiralDataset(untraining_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
holdout_loader = DataLoader(holdout_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
untraining_loader = DataLoader(untraining_dataset, batch_size=64, shuffle=True)

model = Net().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
param_history = []
param_history.append({name: params.clone() for name, params in model.state_dict().items()})

n_epochs = 1000
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

poison_points = []
n_epochs = 10000
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

# converting param_history and poison_points into cpu
param_history = [{name: params.cpu() for name, params in param.items()} for param in param_history]
poison_points = [{name: params.cpu() for name, params in poison.items()} for poison in poison_points]

save_dir = 'params/spiral/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save param_history and poison_points
with open(os.path.join(save_dir, 'param_history.pkl'), 'wb') as f:
    pickle.dump(param_history, f)

with open(os.path.join(save_dir, 'poison_points.pkl'), 'wb') as f:
    pickle.dump(poison_points, f)