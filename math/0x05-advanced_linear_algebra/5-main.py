#!/usr/bin/env python3

if __name__ == '__main__':
    definiteness = __import__('5-definiteness').definiteness
    import numpy as np

    mat1 = np.array([[5, 1], [1, 1]])
    mat2 = np.array([[2, 4], [4, 8]])
    mat3 = np.array([[-1, 1], [1, -1]])
    mat4 = np.array([[-2, 4], [4, -9]])
    mat5 = np.array([[1, 2], [2, 1]])
    mat6 = np.array([])
    mat7 = np.array([[1, 2, 3], [4, 5, 6]])
    mat8 = [[1, 2], [1, 2]]
    mat9 = np.array([[2,1,3,0],[1,0,2,3],[3,2,0,1],[2,0,1,3]])
    mat10 = np.array([[5, 7, 9], [3, 1, 8], [6, 2, 4]])

    print(definiteness(mat1))
    print(definiteness(mat2))
    print(definiteness(mat3))
    print(definiteness(mat4))
    print(definiteness(mat5))
    print(definiteness(mat6))
    print(definiteness(mat7))
    print(definiteness(mat9))
    print(definiteness(mat10))
    try:
        definiteness(mat8)
    except Exception as e:
        print(e)
