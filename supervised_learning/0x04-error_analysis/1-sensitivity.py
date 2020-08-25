#!/usr/bin/env python3
'''sensitivity confusion matrix'''


import numpy as np


def sensitivity(confusion):
    '''calculates the sensitivity for each class in a confusion matrix:

        @confusion: ndarray of shape (classes, classes) where row indices
                    represent the correct labels and column indices represent
                    the predicted labels
            classes is the number of classes
        Returns: ndarray of shape (classes,) containing the sensitivity of
                each class
    '''
    true_pos = np.diag(confusion)
    false_neg = np.sum(confusion, axis=1) - true_pos
    sensitivity = true_pos / (true_pos + false_neg)
    return sensitivity
