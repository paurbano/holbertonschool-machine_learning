#!/usr/bin/env python3


def mat_mul(mat1, mat2):
    ''' matrix multiplication '''
    matrix = []
    m1 = len(mat1)
    n1 = len(mat1[0])
    m2 = len(mat2)
    n2 = len(mat2[0])
    a = [row[:] for row in mat1]
    b = [row[:] for row in mat2]

    if n1 == m2:
        # create new matrix
        for i in range(m1):
            lis = [0] * n2
            matrix.append(lis)

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                for k in range(n1):
                    matrix[i][j] += a[i][k] * b[k][j]
        return matrix
    else:
        return None
