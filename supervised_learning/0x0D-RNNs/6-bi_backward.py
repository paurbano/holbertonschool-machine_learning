#!/usr/bin/env python3
'''Bidirectional Cell Forward'''
import numpy as np


class BidirectionalCell():
    '''class BidirectionalCell'''
    def __init__(self, i, h, o):
        '''Constructor
        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
        Whf and bhfare for the hidden states in the forward direction
        Whb and bhbare for the hidden states in the backward direction
        Wy and byare for the outputs
        '''
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        '''calculates the hidden state in the forward direction for one time
           step
        Args:
            x_t is a numpy.ndarray of shape (m, i) that contains the data
                input for the cell
                m is the batch size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing the previous
                hidden state
        Returns: h_next, the next hidden state
        '''
        h_next = np.tanh(np.dot(np.hstack((h_prev, x_t)), self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        '''calculates the hidden state in the backward direction for one
           time step
        Args:
            x_t is a numpy.ndarray of shape (m, i) that contains the data input
                for the cell
                m is the batch size for the data
            h_next is a numpy.ndarray of shape (m, h) containing the next
                    hidden state
        Returns: h_pev, the previous hidden state
        '''
        h_prev = np.tanh(np.dot(np.hstack((h_next, x_t)), self.Whb) + self.bhb)
        return h_prev
