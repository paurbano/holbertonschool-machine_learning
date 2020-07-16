#! /usr/bin/env python3
''' concatenate two matrices along specific axis '''


def cat_matrices2D(mat1, mat2, axis=0):
    ''' concatenate two matrices along specific axis '''
    # rows matrix 1
    m1 = len(mat1)
    # columns matrix 1
    n1 = len(mat1[0])
    # rows matrix 2
    m2 = len(mat2)
    # columns matrix 2
    n2 = len(mat2[0])
    # new matrix
    matrix = []
    # make a copy of data
    a = [row[:] for row in mat1]
    b = [row[:] for row in mat2]
    if (axis == 1) and (m1 == m2):
        for i in range(m1):
            matrix.append(a[i] + b[i])
    elif (axis == 0) and (n1 == n2):
        matrix = a + b
    else:
        return None
    return matrix
