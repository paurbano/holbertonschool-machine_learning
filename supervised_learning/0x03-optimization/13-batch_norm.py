#!/usr/bin/env python3
''' batch normalization'''

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''normalizes an unactivated output of a neural network using
       batch normalization
       @Z: is a numpy.ndarray of shape (m, n) that should be normalized
        m is the number of data points
        n is the number of features in Z
       @gamma: array of shape (1, n) containing the scales used for
               batch normalization
       @beta: array of shape (1, n) containing the offsets used for
              batch normalization
       @epsilon: is a small number used to avoid division by zero
       Returns: the normalized Z matrix
    '''
    mean = np.sum(Z, axis=0) / Z.shape[0]
    var = np.sum(np.power(Z - mean, 2), axis=0) / Z.shape[0]
    Znorm = (Z - mean) / (np.sqrt(var) + epsilon)
    Zn = gamma * Znorm + beta
    return Zn
