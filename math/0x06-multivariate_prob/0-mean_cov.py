#!/usr/bin/env python3
'''mean and covariance'''

import numpy as np


def mean_cov(X):
    '''calculates the mean and covariance of a data set:
    Args:
        X is a numpy.ndarray of shape (n, d) containing the data set
            n is the number of data points
            d is the number of dimensions in each data point
    Returns: mean, cov:
            mean is a numpy.ndarray of shape (1, d) containing the mean of the
            data set
            cov is a numpy.ndarray of shape (d, d) containing the covariance
            matrix of the data set
    '''
    if len(X.shape) > 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if len(X.shape) < 2:
        raise ValueError('X must contain multiple data points')
    mean = X.mean(axis=0)
    cov = np.dot((X - X.mean(axis=0)).T, (X - X.mean(axis=0))) / X.shape[0]
    return mean.reshape(1, X.shape[1]), cov
