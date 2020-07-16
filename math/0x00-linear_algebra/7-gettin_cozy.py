#! /usr/bin/env python3
''' concatenate two matrices along specific axis '''


def cat_matrices2D(mat1, mat2, axis=0):
    '''concatenate two matrices along specific axis '''
    concat = []
    # make a copy of data
    a = [row[:] for row in mat1]
    b = [row[:] for row in mat2]
    # Know what is the size of the columns and rows
    mat1m = len(mat1)
    mat1n = len(mat1[0])
    mat2m = len(mat2)
    mat2n = len(mat2[0])
    # know waht is the axis that we are working
    if (axis == 1) and (mat1m == mat2m):
        for i in range(mat1m):
            concat.append(a[i] + b[i])
    elif (axis == 0) and (mat1n == mat2n):
        concat = a + b
    else:
        return None
    return concat
