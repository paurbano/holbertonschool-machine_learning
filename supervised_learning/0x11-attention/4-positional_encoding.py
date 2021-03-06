#!/usr/bin/env python3
'''Positional Encoding
https://www.tensorflow.org/tutorials/text/transformer#positional_encoding
'''
import numpy as np


def get_angles(pos, i, d_model):
    '''get_angles'''
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(max_seq_len, dm):
    '''calculates the positional encoding for a transformer
    Args:
        max_seq_len is an integer representing the maximum sequence length
        dm is the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing the
            positional encoding vectors
    '''
    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                            np.arange(dm)[np.newaxis, :],
                            dm)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads
