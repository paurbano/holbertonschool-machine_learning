#!/usr/bin/env python3
'''variance'''

import numpy as np


def variance(X, C):
    '''calculates the total intra-cluster variance for a data set
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the data set
        C: is a numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster
    Returns: var, or None on failure
        var: is the total variance
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    # within variance
    var = np.sum((X - C[:, np.newaxis])**2, axis=-1)
    # between clusters
    mean = np.sqrt(var)
    mini = np.min(mean, axis=0)
    var = np.sum(mini ** 2)
    return np.sum(var)
