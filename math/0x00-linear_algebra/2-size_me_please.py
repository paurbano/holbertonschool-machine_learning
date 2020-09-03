#!/usr/bin/env python3
''' function that calculates the shape of a matrix'''


def matrix_shape(matrix):
    ''' function that calculates the shape of a matrix'''
    shape = []

    while type(matrix) is list:
        '''size of dimension'''
        shape.append(len(matrix))
        '''move to next one'''
        if matrix[0]:
            matrix = matrix[0]
        else:
            break
    return shape
