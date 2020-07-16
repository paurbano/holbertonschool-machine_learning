#! /usr/bin/env python3
''' concatenate two matrices along specific axis '''


def cat_matrices2D(mat1, mat2, axis=0):
    ''' concatenate two matrices along specific axis '''
    # rows matrix 1
    m = len(mat1)
    # columns matrix 1
    n = len(mat1[0])
    # rows matrix 2
    r = len(mat2)
    # columns matrix 2
    c = len(mat2[0])
    # new matrix
    matrix = []
    # make a copy of data
    a = [row[:] for row in mat1]
    b = [row[:] for row in mat2]
    if (axis == 1) and (m == r):
        for i in range(m):
            matrix.append(a[i] + b[i])
    elif (axis == 0) and (n == c):
        matrix = a + b
    else:
        return None
    return matrix
