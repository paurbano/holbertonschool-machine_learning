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
    if Y is None:
        return None
    if classes <= 0 or classes is None:
        return None
    oh = np.zeros((classes, Y.shape[0]))

    for i in range(classes):
        oh[Y[i], i] = 1
    return oh
