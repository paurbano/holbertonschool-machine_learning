#!/usr/bin/env python3
'''transpose a Matrix '''


def matrix_transpose(matrix):
    '''transpose a Matrix '''
    
    # rows
    m = len(matrix)
    # columns
    n = len(matrix[0])
    # new matrix
    transpose = []

    for i in range(n):
        l = [0] * m
        transpose.append(l)

    for i in range(m):
        for j in range(n):
            transpose[j][i] = matrix[i][j]
    return transpose
