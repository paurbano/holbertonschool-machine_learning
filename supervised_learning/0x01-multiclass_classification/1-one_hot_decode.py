#!/usr/bin/env python3
'''Multiclass Classification '''

import numpy as np


def one_hot_decode(one_hot):
    '''one-hot matrix into a vector of labels
        one_hot: is a one-hot encoded array with shape (classes, m)
            classes: is the maximum number of classes
            m: is the number of examples
        Returns: array shape (m, ) with the numeric labels for each example
    '''
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot) == 0:
        return None
    if len(one_hot.shape) != 2:
        return None
    one_decode = np.zeros((one_hot.shape[1], ), dtype=int)
    row = 0
    for arr in one_hot:
        pos = np.where(arr == 1)
        one_decode[pos] = int(row)
        row += 1
    return one_decode
