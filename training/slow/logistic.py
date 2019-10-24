import numpy as np
import matplotlib.pyplot as plt
from util.util import readFile, sigmoid

def train(nTimes, rate):

    data, n, m = readFile(fn, mode, logistic=True)

    features = data[:, :-1]  # array of training examples
    labels = data[:, -1]  # array of corresponding labels
    thetas = np.zeros(m)  # parameters
    gradient = np.zeros(m)  # gradient

    for k in range(nTimes):
        gradient = np.zeros(m)
        for i in range(n):
            y = labels[i]
            x = features[i]
            z = np.dot(thetas.transpose(), x)
            ascent = y - sigmoid(z)
            gradient += x * ascent
        thetas += rate * gradient

    return thetas