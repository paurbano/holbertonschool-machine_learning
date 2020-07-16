#!/usr/bin/env python3
''' function that calculates the shape of a matrix'''


def matrix_shape(matrix):
    ''' function that calculates the shape of a matrix'''
    shape = []

    if not isinstance(matrix, list):
        return None

    if matrix is None or len(matrix) == 0:
        return shape

    if isinstance(matrix[0], list):
        shape.append(len(matrix))
        shape.append(len(matrix[0]))
        if isinstance(matrix[0][0], list):
            shape.append(len(matrix[0][0]))
        return shape
    elif isinstance(matrix[0], int):
        shape.append(len(matrix))
        return shape
    else:
        return None
