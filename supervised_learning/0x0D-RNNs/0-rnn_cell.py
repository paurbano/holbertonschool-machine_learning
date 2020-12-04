#!/usr/bin/env python3
'''class RNNCell that represents a cell of a simple RNN'''
import numpy as np


class RNNCell ():
    '''
    '''
    def __init__(self, i, h, o):
        '''constructor
        Args:
            -i is the dimensionality of the data
            -h is the dimensionality of the hidden state
            -o is the dimensionality of the outputs
            Creates the public instance attributes Wh, Wy, bh, by that
            represent the weights and biases of the cell
                Wh and bh are for the concatenated hidden state and input data
                Wy and by are for the output
        '''
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """ Softmax function """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        '''performs forward propagation for one time step
        Args:
            x_t: is a numpy.ndarray of shape (m, i) that contains the data
                input for the cell
                m is the batche size for the data
            h_prev: is a numpy.ndarray of shape (m, h) containing the previous
                    hidden state
        Returns: h_next, y
                h_next is the next hidden state
                y is the output of the cell
        '''
        h_next = np.tanh(np.dot(np.hstack((h_prev, x_t)), self.Wh) + self.bh)
        y = np.dot(h_next, self.Wy) + self.by
        return h_next, self.softmax(y)
