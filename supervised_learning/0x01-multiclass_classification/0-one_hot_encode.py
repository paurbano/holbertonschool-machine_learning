#!/usr/bin/env python3
'''One-Hot Encode'''


import numpy as np


def one_hot_encode(Y, classes):
    '''converts a numeric label vector into a one-hot matrix
        Y: array with shape (m,) containing numeric class labels
            m: is the number of examples
        classes: is the maximum number of classes found in Y
        Returns: a one-hot encoding of Y with shape (classes, m)
    '''
    if type(Y) is not np.ndarray:
        return None
    if len(Y) == 0:
        return None
    if type(classes) is not int or classes <= Y.max():
        return None

    one_hot = np.zeros((classes, Y.shape[0]))

    for i in range(len(Y)):
        one_hot[Y[i]][i] = 1
    return one_hot
