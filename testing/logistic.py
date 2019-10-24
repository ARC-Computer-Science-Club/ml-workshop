# Implementation of Logistic Regression algorithm for CS109 @ Stanford
# Ⓒ Artem Tkachuk

import numpy as np
from util.readFile import readFile
from util.vdecider import vdecider
from scipy.special import expit

def test(fn, thetas):

    fileName = f'data/test/{fn}.test'
    tests, n, _ = readFile(fileName, logistic=True)

    features = tests[:, :-1]    # array of training examples
    labels = tests[:, -1]       # array of corresponding labels
    values = np.unique(labels)  # all possible values the labels have

    p = expit(np.matmul(features, thetas.transpose()))
    y_hats = vdecider(p)

    total_zero, guessed_zero, total_one, guessed_one  = 0, 0, 0, 0

    #Testing
    for label, y_hat in zip(labels, y_hats):
        if label == 1:
            total_one += 1
            if y_hat == 1:
                guessed_one += 1
        elif label == 0:            # could use just else, but this way it's more clear
            total_zero += 1
            if y_hat == 0:
                guessed_zero += 1

    total_guessed =  guessed_zero + guessed_one

    #Preparing report
    report = f'For "{fn}" dataset:\n'
    report += f'Class 0: tested {total_zero}, Correctly classified: {guessed_zero}\n'
    report += f'Class 1: tested {total_one}, Correctly classified: {guessed_one}\n'
    report += f'Overall: tested {n}, correctly classified {total_guessed}\n'
    report += f'Accuracy: {float(total_guessed) / n}\n\n'

    of = open('results/logistic.txt', 'a+')
    of.write(report)
    # print(report)     # can also write to console