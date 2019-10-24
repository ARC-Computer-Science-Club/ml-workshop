'''
    Implementation of a simple neural network with 1 hidden layer
    Ⓒ Artem Tkachuk
'''

from training.slowWithLoops.neural import train
from testing.neural import test

def neural():

    datasets = [
        {
            'name': 'SPECT',
            'nTimes': 3000,
            'rate': 0.001
        },
        # You can add another datasets here
    ]

    mh = 13  # number of neurons in the hidden layer

    for ds in datasets:
        thetas = train(ds['name'], ds['nTimes'], ds['rate'], mh)
        test(ds['name'], thetas)