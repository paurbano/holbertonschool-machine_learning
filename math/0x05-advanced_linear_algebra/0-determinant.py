#!/usr/bin/env python3
'''calculates the determinant of a matrix'''


def determinant(matrix):
    '''calculates the determinant of a matrix
    Args:
        matrix:  is a list of lists whose determinant should be calculated
    Returns: the determinant of matrix
    '''
    if type(matrix) is not list or not matrix:
        raise TypeError('matrix must be a list of lists')
    for data in matrix:
        if type(data) is not list:
            raise TypeError('matrix must be a list of lists')
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError('matrix must be a square matrix')
    # length of matrix and makes a copy
    n = len(matrix)
    AM = matrix[:]

    # Section 2: Row manipulate A into an upper triangle matrix
    for fd in range(n):  # fd stands for focus diagonal
        if AM[fd][fd] == 0:
            AM[fd][fd] = 1.0e-18  # Cheating by adding zero + ~zero
        for i in range(fd+1, n):  # skip row with fd in it.
            crScaler = AM[i][fd] / AM[fd][fd]  # cr stands for "current row".
            for j in range(n):  # cr - crScaler * fdRow, one element at a time.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]

    # Section 3: Once AM is in upper triangle form ...
    product = 1
    for i in range(n):
        product *= AM[i][i]  # ... product of diagonals is determinant

    return round(product)
