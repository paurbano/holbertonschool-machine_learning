#!/usr/bin/env python3
'''precision confusion matrix'''


import numpy as np


def precision(confusion):
    '''calculates the precision for each class in a confusion matrix:

        @confusion: ndarray of shape (classes, classes) where row indices
                    represent the correct labels and column indices represent
                    the predicted labels
            classes is the number of classes
        Returns: ndarray of shape (classes,) containing the precision of
                each class
    '''
    true_pos = np.diag(confusion)
    false_pos = np.sum(confusion, axis=0) - true_pos
    precision = true_pos / (true_pos + false_pos)
    return precision
