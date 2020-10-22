#!/usr/bin/env python3
'''cofactor matrix of a matrix'''


def adjugate(matrix):
    '''calculates the adjugate matrix of a matrix:
    Args:
        matrix: is a list of lists whose adjugate matrix should be calculated
    Return:
        the adjugate matrix of matrix
    '''
    if not matrix:
        raise TypeError('matrix must be a list of lists')
    for data in matrix:
        if type(data) is not list:
            raise TypeError('matrix must be a list of lists')
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError('matrix must be a non-empty square matrix')
    # Order of the matrix
    m = len(matrix[0])  # np.shape(A)[0]
    # Initializing the cofactor matrix with zeros
    cofactor = zeros_matrix(m, m)
    for i in range(m):
        for j in range(m):
            # print(getMatrixMinor(matrix, i, j))
            cofactor[i][j] = pow(-1, i+j) * determinant(submat(matrix, i, j))
    adjugate = matrix_transpose(cofactor)
    return adjugate


def zeros_matrix(rows, cols):
    """Creates a matrix filled with zeros.
    Args:
        rows: the number of rows the matrix should have
        cols: the number of columns the matrix should have
    return: list of lists that form the matrix
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0)

    return M


def determinant(matrix):
    '''calculates the determinant of a matrix
    Args:
        matrix:  is a list of lists whose determinant should be calculated
    Returns: the determinant of matrix
    '''
    # if type(matrix) is not list or not matrix:
    #    raise TypeError('matrix must be a list of lists')
    for data in matrix:
        if type(data) is not list:
            raise TypeError('matrix must be a list of lists')
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError('matrix must be a square matrix')
    # Section 1: Establish n parameter and copy A
    n = len(matrix)
    AM = matrix[:]  # copy_matrix(matrix)

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


def submat(M, row, col):
    '''Create submatrix for minor, deleting the row and column
    Args:
        M: Original Matrix
        row: row to delete
        col: column to delete
    Return: Matrix without row and column
    '''
    minor = []
    rows = len(M)
    cols = len(M[0])
    for i in range(rows):
        if i == row:
            continue
        colu = []
        for j in range(cols):
            if j == col:
                continue
            colu.append(M[i][j])
        minor.append(colu)
    # print(minor)
    return minor


def matrix_transpose(matrix):
    '''transpose a Matrix '''

    # rows
    m = len(matrix)
    # columns
    n = len(matrix[0])
    # new matrix
    transpose = []

    for i in range(n):
        lis = [0] * m
        transpose.append(lis)

    for i in range(m):
        for j in range(n):
            transpose[j][i] = matrix[i][j]
    return transpose
