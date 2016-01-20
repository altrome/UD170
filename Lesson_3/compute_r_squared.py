import numpy as np

def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and 'predictions,'
    # returns the coefficient of determination, R^2, for the model that produced 
    # predictions.
    # 
    # Numpy has a couple of functions -- np.mean() and np.sum() --
    # that you might find useful, but you don't have to use them.

    data_mean = np.mean(data)
    den = []
    num = []
    den[:] = [(x - data_mean)**2 for x in data]
    num[:] = [(i - j)**2 for i, j in zip(data, predictions)]
    n = np.sum(num)
    d = np.sum(den)        
    r_squared = 1 - (n/d)
    return r_squared
