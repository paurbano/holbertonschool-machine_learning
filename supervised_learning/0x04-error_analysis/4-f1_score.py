#!/usr/bin/env python3
'''F1 score confusion matrix'''


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    '''calculates the F1 score of a confusion matrix:

        @confusion: ndarray of shape (classes, classes) where row indices
                    represent the correct labels and column indices represent
                    the predicted labels
            classes is the number of classes
        Returns: ndarray of shape (classes,) containing the F1 score of
                each class
    '''
    sensi = sensitivity(confusion)
    preci = precision(confusion)
    f1_score = 2 * ((preci * sensi) / (preci + sensi))
    return f1_score
