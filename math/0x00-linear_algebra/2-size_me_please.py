#!/usr/bin/env python3
''' shape of a matrix'''
def matrix_shape(matrix):
    ''' function that calculates the shape of a matrix'''
    shape = []
    if matrix is None or len(matrix) == 0:
        return shape
    
    shape.append(len(matrix))
    shape.append(len(matrix[0]))
    if isinstance(matrix[0][0], list):
        shape.append(len(matrix[0][0]))
    return shape
