#!/usr/bin/env python3
'''Bidirectional RNN'''

import numpy as np

def bi_rnn(bi_cell, X, h_0, h_t):
    '''performs forward propagation for a bidirectional RNN
    Args:
        bi_cell is an instance of BidirectinalCell that will be used for the
                forward propagation
        X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
            t is the maximum number of time steps
            m is the batch size
            i is the dimensionality of the data
        h_0 is the initial hidden state in the forward direction, given as a
            numpy.ndarray of shape (m, h)
            h is the dimensionality of the hidden state
        h_t is the initial hidden state in the backward direction, given as a
            numpy.ndarray of shape (m, h)
    Returns: H, Y
        H is a numpy.ndarray containing all of the concatenated hidden states
        Y is a numpy.ndarray containing all of the outputs
    '''
    T = X.shape[0]
    Hf = []
    Hb = []
    h_prev = h_0
    h_next = h_t
    for t in range(T):
        h_prev = bi_cell.forward(h_prev, X[t])
        h_next = bi_cell.backward(h_next, X[T - 1 - t])
        Hf.append(h_prev)
        Hb.append(h_next)
    Hb = [x for x in reversed(Hb)]
    Hf = np.array(Hf)
    Hb = np.array(Hb)
    H = np.concatenate((Hf, Hb), axis=-1)
    Y = bi_cell.output(H)
    return (H, Y)
