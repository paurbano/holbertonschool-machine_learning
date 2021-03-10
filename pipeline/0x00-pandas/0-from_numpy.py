#!/usr/bin/env python3
'''creates a pd.DataFrame from a np.ndarray'''
import pandas as pd


def from_numpy(array):
    '''creates a pd.DataFrame from a np.ndarray
    Args:
        array is the np.ndarray from which you should create the pd.DataFrame
    Returns: the newly created pd.DataFrame
    '''
    indexing = [chr(i+65) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=indexing)
