#!/usr/bin/env python3
'''Specificity'''


import numpy as np


def specificity(confusion):
    ''' calculates the specificity for each class in a confusion matrix:

    @confusion: a confusion numpy.ndarray of shape (classes, classes)
                where row indices represent the correct labels and column
                indices represent the predicted labels
        classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the specificity
    '''
    true_pos = np.diag(confusion)
    false_pos = np.sum(confusion, axis=0) - true_pos
    false_ne = np.sum(confusion, axis=1) - true_pos
    true_neg = np.sum(confusion) - (true_pos + false_pos + false_ne)
    specificity = true_neg / (true_neg + false_pos)
    return specificity
