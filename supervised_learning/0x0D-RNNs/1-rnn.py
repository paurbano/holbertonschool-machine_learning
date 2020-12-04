#!/usr/bin/env python3
'''performs forward propagation for a simple RNN'''
import numpy as np


def rnn(rnn_cell, X, h_0):
    '''forward propagation for a simple RNN
    Args:
        rnn_cell: is an instance of RNNCell that will be used for the forward
                propagation
        X: is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0: is the initial hidden state, given as a numpy.ndarray of shape
            (m, h) h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    '''
    T = X.shape[0]
    H = []
    Y = []
    h = h_0
    H.append(h)
    for t in range(T):
        h, y = rnn_cell.forward(h, X[t])
        H.append(h)
        Y.append(y)
    H = np.array(H)
    Y = np.array(Y)
    return (H, Y)
