#!/usr/bin/env python3
'''normalizes a matrix'''

import numpy as np


def normalize(X, m, s):
    ''' normalize a matrix
        X is the numpy.ndarray of shape (d, nx) to normalize
            d: is the number of data points
            nx: is the number of features
        m: array of shape (nx,) that contains the mean of all features of X
        s: array of shape (nx,) that contains the standard deviation
           of all features of X
        Returns: The normalized X matrix
    '''
    return (X - m) / s
