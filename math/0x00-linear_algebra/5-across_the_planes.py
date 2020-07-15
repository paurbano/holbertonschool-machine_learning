#!/usr/bin/env python3


def add_matrices2D(mat1, mat2):
    ''' sum two matrices '''
    # rows matrix 1
    m1 = len(mat1)
    # rows matrix 2
    m2 = len(mat2)
    # columns matrix 1
    n1 = len(mat1[0])
    # columns matrix 2
    n2 = len(mat2[0])
    # new matrix
    _sum = []
    if m1 == m2 and n1 == n2:
        # create new matrix with size m x n
        for i in range(m1):
            lis = [0] * n1
            _sum.append(lis)

        for i in range(m1):
            for j in range(n1):
                _sum[i][j] = mat1[i][j] + mat2[i][j]
        return _sum
    else:
        return None
