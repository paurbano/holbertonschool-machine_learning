#!/usr/bin/env python3
'''class MultiNormal'''

import numpy as np


class MultiNormal():
    '''class MultiNormal that represents a Multivariate Normal distribution
    '''
    def __init__(self, data):
        '''data is a numpy.ndarray of shape (d, n) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point
        '''
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[0] < 2:
            raise ValueError('data must contain multiple data points')
        n = data.shape[1]
        self.mean = data.mean(axis=1, keepdims=True)
        self.cov = np.dot((data - self.mean), (data - self.mean).T) / (n - 1)
