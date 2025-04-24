import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Simple MLP model with 3 hidden layers
    input_dim: int, input dimension
    output_dim: int, output dimension
    """
    def __init__(self, input_dim = 1, output_dim = 1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.output = nn.Linear(10, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.output(x)
        return x
        