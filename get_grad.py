import sys
import torch
import torch.nn as nn
import os
import numpy as np

def get_val_and_grad(x):


    net = torch.load(os.path.join(os.getcwd(), 'saved_models/model.pt'))
    # x = torch.tensor(x.reshape((1, 6)).tolist(), requires_grad=True)
    x = torch.tensor(np.array(x).reshape(1, 6).tolist(), requires_grad=True)
    # x = torch.ones(1, 6, requires_grad=True)
    out = net(x)
    out.backward()
    print(x.grad)

    return x.grad.detach().numpy().tolist(), out.detach().numpy().tolist()

class Network(nn.Module):
    def __init__(self, net_dims):
        """Constructor for multi-layer perceptron pytorch class

        params:
            net_dims: list of ints  - dimensions of each layer in neural network
        """

        super(Network, self).__init__()

        layers = []
        for i in range(len(net_dims) - 1):

            if isinstance(net_dims[i + 1], str):
                layers.append(nn.Linear(net_dims[i], net_dims[i + 2]))
            else:
                layers.append(nn.Linear(net_dims[i], net_dims[i + 1]))

            # use activation function if not at end of layer
            if i != len(net_dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Pass data through the network model

        params:
            x: torch tensor - data to pass though neural network
        """

        return torch.squeeze(self.net(x))

if __name__ == '__main__':
    squared()
