import torch.nn as nn
import torch

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
