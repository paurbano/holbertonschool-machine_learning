#!/usr/bin/env python3
'''concatenates two matrices'''


import numpy as np


def np_cat(mat1, mat2, axis=0):
    ''' concatenates two matrices'''
    matrix = np.concatenate((mat1, mat2), axis)
    return matrix
