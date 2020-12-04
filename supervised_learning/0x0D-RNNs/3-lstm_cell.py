#!/usr/bin/env python3
'''LSTM Cell'''
import numpy as np


class LSTMCell():
    '''class LSTM Cell'''
    def __init__(self, i, h, o):
        '''Constructor
        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
        Wf and bf are for the forget gate
        Wu and bu are for the update gate
        Wc and bc are for the intermediate cell state
        Wo and bo are for the output gate
        Wy and by are for the outputs
        '''
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, h))

    def softmax(self, x):
        """ Softmax function """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Sigmoid"""
        return (1 / (1 + np.exp(-x)))

    def forward(self, h_prev, c_prev, x_t):
        '''performs forward propagation for one time step
        Args:
            x_t is a numpy.ndarray of shape (m, i) that contains the data input
                for the cell
                m is the batche size for the data
            h_prev is a numpy.ndarray of shape (m, h) containing the previous
                    hidden state
            c_prev is a numpy.ndarray of shape (m, h) containing the previous
                    cell state
        Returns: h_next, c_next, y
                h_next is the next hidden state
                c_next is the next cell state
                y is the output of the cell
        '''
        U = np.hstack((h_prev, x_t))
        f = self.sigmoid(np.dot(U, self.Wf) + self.bf)
        u = self.sigmoid(np.dot(U, self.Wu) + self.bu)
        c_bar = np.tanh(np.dot(U, self.Wc) + self.bc)
        c_next = f * c_prev + u * c_bar
        o = self.sigmoid(np.dot(U, self.Wo) + self.bo)
        h_next = o * np.tanh(c_next)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return (h_next, c_next, y)
