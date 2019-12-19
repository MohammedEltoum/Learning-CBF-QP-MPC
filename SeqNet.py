import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_hidden, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_hidden)
        self.fc2 = nn.Linear(num_hidden, output_dim)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)

        return torch.squeeze(X)
