'''
    Implementation of Logistic Regression algorithm
    Ⓒ Artem Tkachuk
'''

import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt;  plt.ion()
from util.readFile import readFile
from util.sigmoid import sigmoid
from util.logLikelihood import logLikelihood
from graphing.graph import init, replot

def train(fn, nTimes, rate):
    fileName = f'data/train/{fn}.train'
    data, n, m = readFile(fileName)

    features = data[:, :-1]  # array of training examples
    labels = data[:, -1]  # array of corresponding labels
    thetas = np.zeros(m)  # parameters
    gradient = np.zeros(m)  # gradient
    y_hats = np.zeros(labels.shape)

    fig, ax, xdata, ydata, line = init(fn)

    for k in range(nTimes):
        gradient = np.zeros(m)
        for i in range(n):
            y = labels[i]
            x = features[i]
            z = np.dot(thetas.transpose(), x)
            y_hat = sigmoid(z)
            delta = y - y_hat
            gradient += x * delta
            y_hats[i] = y_hat

        thetas += rate * gradient

        LL = logLikelihood(labels, y_hats)
        replot(fig, ax, line, nTimes, xdata, ydata, k, LL)

    plt.savefig(f'graphing/pics/logistic/{fn}-slow.png')

    return thetas