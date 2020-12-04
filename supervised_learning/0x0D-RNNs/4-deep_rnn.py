#!/usr/bin/env python3
'''forward propagation for a deep RNN'''
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    '''forward propagation for a deep RNN
    Args:
        rnn_cells is a list of RNNCell instances of length l that will be used
                for the forward propagation
            l is the number of layers
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state, given as a numpy.ndarray of shape
            (l, m, h)
            h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    '''
    T = X.shape[0]
    H = [[h for h in h_0]]
    Y = []
    for t in range(T):
        Htemp = []
        x_t = X[t]
        for i, cell in enumerate(rnn_cells):
            h, y = cell.forward(H[t][i], x_t)
            Htemp.append(h)
            x_t = h
        H.append(Htemp)
        Y.append(y)
    H = np.array(H)
    Y = np.array(Y)
    return (H, Y)
