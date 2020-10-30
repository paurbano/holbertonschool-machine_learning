#!/usr/bin/env python3
'''pca V2 on a dataset'''

import numpy as np


def def pca(X, ndim):
    '''performs PCA on a dataset:
    Args:
        X: is a numpy.ndarray of shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each point
            all dimensions have a mean of 0 across all data points
        ndim: is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim) containing the
            transformed version of X
    '''
    """Function that performs PCA on a dataset"""
    normal = np.mean(X, axis=0)
    X_normal = X - normal
    vh = np.linalg.svd(X_normal)[2]
    Weights_r = vh[: ndim].T
    T = np.matmul(X_normal, Weights_r)

    return T
