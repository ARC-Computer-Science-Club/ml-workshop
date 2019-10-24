'''
    Implementation of logistic regression algorithm
    Ⓒ Artem Tkachuk
'''

from training.slow.logistic import train
#from training.logistic import train
from testing.logistic import test

def logistic():

    datasets = [
        {
            'name': 'SPECT',
            'nTimes': 10000,
            'rate': 0.001
        },
        # You can add another datasets here
    ]

    for ds in datasets:
        thetas = train(ds['name'], ds['nTimes'], ds['rate'])
        test(ds['name'], thetas)