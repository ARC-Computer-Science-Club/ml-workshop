'''
    Implementation of a simple neural network with 1 hidden layer
    Ⓒ Artem Tkachuk
'''

# TODO refactor this code
#TODO https://stackoverflow.com/questions/14290113/git-pushing-code-to-two-remotes

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt;  plt.ion()
import numpy as np
import math
from scipy.special import expit
from util.readFile import readFile
from util.logLikelihood import logLikelihood
from util.sigmoid import sigmoid
from graphing.graph import init, replot

def train(fn, nTimes, rate, mh):

    fileName = f'data/train/{fn}.train'
    data, n, mx = readFile(fileName)     # of training examples and neurons in the hidden layer

    features = data[:, :-1]              # matrix of training examples
    labels = data[:, -1]                 #vector of corresponding labels
    y_hats = np.empty(labels.shape)      #vector of predictions

    thetas_h = np.random.normal(0, math.sqrt(1 / mx), ((mx, mh)))    #parameters for hidden layer
    thetas_y_hat = np.random.normal(0, math.sqrt(1 / mh), (mh))      #parameters for hidden layer


    fig, ax, xdata, ydata, line = init(fn)     #plotting the log likelihood while training

    for k in range(nTimes):
        gradient_h = np.full((mx, mh), 1 / mx)
        gradient_y_hat = np.full((mh), 1 / mh)

        for example in range(n):
            # Forward Pass, computing hidden layer
            x, y, h = features[example], labels[example], np.empty((mh))

            for j in range(mh):
                sum = 0.0
                for i in range(mx):
                    sum += x[i] * thetas_h[i][j]
                h[j] = sigmoid(sum)

            #computing prediction
            sum = 0
            for j in range(mh):
                sum += thetas_y_hat[j] * h[j]
            y_hat = sigmoid(sum)
            y_hats[example] = y_hat
            delta = y - y_hat

            # computing gradients
            for j in range(mh):
                gradient_y_hat[j] += delta * h[j]

            for i in range(mx):
                for j in range(mh):
                    gradient_h[i][j] += delta * h[j] * (1 - h[j]) * thetas_y_hat[j] * x[i]

        # updating parameters
        for j in range(mh):
            thetas_y_hat[j] += rate * gradient_y_hat[j]

        for i in range(mx):
            for j in range(mh):
                thetas_h[i][j] += rate * gradient_h[i][j]

        LL = logLikelihood(labels, y_hats)
        # print(LL)
        replot(fig, ax, line, nTimes, xdata, ydata, k, LL)


    plt.savefig(f'graphing/pics/neural/{fn}.png')

    return (thetas_h, thetas_y_hat, mh)