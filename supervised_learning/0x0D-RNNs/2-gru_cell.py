#!/usr/bin/env python3
'''GRU Cell'''
import numpy as np


class GRUCell():
    '''GRUCell'''
    def __init__(self, i, h, o):
        '''Constructor
        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
        Wz and bz are for the update gate
        Wr and br are for the reset gate
        Wh and bh are for the intermediate hidden state
        Wy and by are for the output
        '''
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """ Softmax function """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Sigmoid"""
        return (1 / (1 + np.exp(-x)))

    def forward(self, h_prev, x_t):
        '''forward propagation for one time step
        Args:
            x_t is a numpy.ndarray of shape (m, i) that contains the data
                input for the cell
                m is the batche size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing the
                    previous hidden state
        Returns: h_next, y
                h_next is the next hidden state
                y is the output of the cell
        '''
        U = np.hstack((h_prev, x_t))
        z = self.sigmoid(np.dot(U, self.Wz) + self.bz)
        r = self.sigmoid(np.dot(U, self.Wr) + self.br)
        U = np.hstack((h_prev * r, x_t))
        c = np.tanh(np.dot(U, self.Wh) + self.bh)
        h_next = np.multiply(z, c) + np.multiply((1 - z), h_prev)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return (h_next, y)
