#! /usr/bin/env python3
''' '''

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
    if axis == 1 and m == r:
        l = []
        for i in range(m):
            matrix.append(mat1[i] + mat2[i])
    elif axis == 0 and m ==r:
        matrix = mat1 + mat2
    else:
        return None
    return matrix
