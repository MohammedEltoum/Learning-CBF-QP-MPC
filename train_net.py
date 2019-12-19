import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from SeqNet import Net
from MultiLayerNet import Network
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from process_data import get_states_and_values
import os

NUM_HIDDEN = 50
NUM_EPOCHS = 1000
INPUT_DIM = 6
OUTPUT_DIM = 1
LEARNING_RATE = 1e-3

def trial():

    states, values = get_states_and_values()
    train_X, test_X, train_y, test_y = split_dataset(states, values)

    net = Net(NUM_HIDDEN, INPUT_DIM, OUTPUT_DIM)
    # net_dims = [INPUT_DIM, 20, 10, OUTPUT_DIM]
    # net = Network(net_dims)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)

    train_errors = train_network(train_X, train_y, net, optimizer, criterion)

    # torch.save(net, os.path.join(os.getcwd(), 'saved_models/model.pt'))

    return net, train_errors


def split_dataset(states: np.ndarray, values: np.ndarray):
    """Create training/test split and store in torch Variables

    params:
        * states - states of CBF
        * values - values of CBF
    """

    train_X, test_X, train_y, test_y = train_test_split(states, values, test_size=0.2)

    train_X = Variable(torch.Tensor(train_X).float())
    test_X = Variable(torch.Tensor(test_X).float())
    train_y = Variable(torch.Tensor(train_y).float())
    test_y = Variable(torch.Tensor(test_y).float())

    return train_X, test_X, train_y, test_y

def train_network(train_X: torch.tensor, train_y: torch.tensor, net, optimizer, criterion):
    """Train neural network on Iris dataset

    params:
        train_X: training instances
        train_y: training labels
        net: torch nn.Module - neural network model
        optimizer: nn.optim - optimizer for training network
        criterion: nn.CrossEntropyLoss - loss function
    """

    errors = []
    for epoch in tqdm(range(NUM_EPOCHS)):

        out = net(train_X)
        loss = criterion(out, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        err = test_network(train_X, train_y, net, criterion)
        errors.append(err)

    return errors

def test_network(test_X, test_y, net, criterion) -> float:
    """Test neural network trained on Iris dataset

    params:
        test_X: torch tensor - test instances
        test_y: torch tensor - test labels
        net: torch nn.Module - neural network model
        criterion:

    returns:
        error: trained network error on test data
    """

    predict_out = net(test_X)
    return criterion(predict_out, test_y).detach().numpy()


if __name__ == '__main__':
    trial()
