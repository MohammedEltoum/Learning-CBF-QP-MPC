import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from train_net import trial
from process_data import get_states_and_values
from torch.autograd import Variable
import torch
from scipy.io import savemat

from SeqNet import Net
from MultiLayerNet import Network
import os

def main():

    states, values = get_states_and_values()
    net, errors = trial()
    data = Variable(torch.tensor(states).float())
    predicted_values = net(data).detach().numpy()

    # matlab_arr = savemat('data/data_for_haimin.m', {
    #     'X': states,
    #     'y_pred': predicted_values,
    # })

    plot3d(states, values, predicted_values)


    data_pt = states[0,:]
    # print(states[0,:])
    #
    calc_grad(net, data_pt)
    #
    # new_net = torch.load(os.path.join(os.getcwd(), 'saved_models/model.pt'))
    # new_net.eval()
    # calc_grad(new_net, data_pt)



def plot3d(states, values, predicted_values):
    rel_x_dist = states[:,0] - states[:,3]
    rel_y_dist = states[:,1] - states[:,4]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter3D(rel_x_dist, rel_y_dist, predicted_values, c=predicted_values, cmap='Blues')
    ax.scatter3D(rel_x_dist, rel_y_dist, values, c=values, cmap='Reds', alpha=0.2)

    plt.show()


def plot_train_errors(errors):
    plt.plot(range(len(errors)), errors, label='Training error')
    plt.legend()

def calc_grad(net, x):

    x = torch.tensor(x.reshape((1, 6)).tolist(), requires_grad=True)
    out = net(x)
    out.backward()
    print(x.grad)

    return x.grad


if __name__ == '__main__':
    main()
