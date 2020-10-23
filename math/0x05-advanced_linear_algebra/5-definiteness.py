#!/usr/bin/env python3
'''calculates the definiteness of a matrix'''

import numpy as np


def definiteness(matrix):
    '''calculates the definiteness of a matrix
    Args:
        matrix: numpy.ndarray of shape (n, n)
    Return:
        the string Positive definite, Positive semi-definite,
        Negative semi-definite, Negative definite, or Indefinite
    '''
    if type(matrix) is not np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')
    if not matrix.any():
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.array_equal(matrix, matrix.T):
        return None

    eg, _ = np.linalg.eig(matrix)

    if all(eg > 0):
        return "Positive definite"
    if all(eg >= 0):
        return "Positive semi-definite"
    if all(eg < 0):
        return "Negative definite"
    if all(eg <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
