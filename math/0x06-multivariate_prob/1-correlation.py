#!/usr/bin/env python3
'''correlation matrix'''

import numpy as np


def correlation(C):
    '''calculates a correlation matrix
    args:
        C: is a numpy.ndarray of shape (d, d) containing a covariance matrix
            d is the number of dimensions
    Return:
        numpy.ndarray of shape (d, d) containing the correlation matrix
    '''
    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    if C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')
    v = np.sqrt(np.diag(C))
    outer_v = np.outer(v, v)
    correlation = C / outer_v
    correlation[C == 0] = 0
    return correlation
