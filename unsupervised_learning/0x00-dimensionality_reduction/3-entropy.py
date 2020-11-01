#!/usr/bin/env python3
''' Shannon entropy
https://onestopdataanalysis.com/shannon-entropy/
https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
'''

import numpy as np


def HP(Di, beta):
    '''calculates the Shannon entropy and P affinities
    Args:
        Di: is a numpy.ndarray of shape (n - 1,) containing the pariwise
        distances between a data point and all other points except itself
            n is the number of data points
        beta: is a numpy.ndarray of shape (1,) containing the beta value for
            the Gaussian distribution
    Returns: (Hi, Pi)
            Hi: the Shannon entropy of the points
            Pi: a numpy.ndarray of shape (n - 1,) containing the P affinities
                of the points
    '''
    Pi = np.exp(-Di * beta)
    sumPi = np.sum(Pi)
    Pi = Pi / sumPi
    Hi = -np.sum(Pi * np.log2(Pi))
    return Hi, Pi
