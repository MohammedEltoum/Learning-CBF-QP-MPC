import numpy as np
from scipy.io import loadmat
import os

DATA_DIR = 'data'

def get_states_and_values():
    """Load states and values from Haimin's data file

    returns:
        * states - states of CBF
        * values - values of CBF
    """

    X = load_data('X')
    y = load_data('y')

    states = X[:, :6]
    values = y[:, 0]

    test_sizes(states, values)

    return states, values


def load_data(name: str) -> np.ndarray:
    """Load data from MATLAB file and store as np array

    params:
        * name: .mat filename of MATLAB array

    returns:
        * numpy array of datapoints
    """
    data = loadmat(os.path.join(DATA_DIR, name))
    return np.array(data[name])

def test_sizes(arr1: np.ndarray, arr2: np.ndarray):
    """Assert that there are the same number of elements in two arrays

    params:
        * arr1, arr2
    """

    assert(arr1.shape[0] == arr2.shape[0])

if __name__ == '__main__':
    get_states_and_values()
