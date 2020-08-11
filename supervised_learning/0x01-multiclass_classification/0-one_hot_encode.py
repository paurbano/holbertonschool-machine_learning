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
    if len(Y) == 0 or type(classes) is not int:
        return None
    if classes <= max(Y):
        return None
    if not isinstance(Y, np.ndarray):
        return None

    oh = np.zeros((classes, Y.shape[0]))

    for i in range(classes):
        oh[Y[i], i] = 1
    return oh
