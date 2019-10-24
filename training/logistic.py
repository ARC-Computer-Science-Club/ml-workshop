'''
    Implementation of Logistic Regression algorithm
    Ⓒ Artem Tkachuk
'''

import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt;  plt.ion()
from scipy.special import expit
from util.readFile import readFile
from util.logLikelihood import logLikelihood
from graphing.graph import init, replot


def train(fn, nTimes, rate):
    fileName = f'data/train/{fn}.train'
    data, n, m = readFile(fileName)

    features = data[:, :-1]  # array of training examples
    labels = data[:, -1]  # array of corresponding labels
    thetas = np.zeros(m)  # parameters
    gradient = np.zeros(m)  # vector of gradients

    fig, ax, xdata, ydata, line = init(fn)

    for k in range(nTimes):
        y_hats = expit(np.matmul(features, thetas.transpose()))
        gradient = np.sum(features * (labels - y_hats)[:, np.newaxis], axis=0)
        thetas += rate * gradient

        LL = logLikelihood(labels, y_hats)
        replot(fig, ax, line, nTimes, xdata, ydata, k, LL)

    plt.savefig(f'graphing/pics/logistic/{fn}.png')

    return thetas