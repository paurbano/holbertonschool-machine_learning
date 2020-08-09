#!/usr/bin/env python3
'''class DeepNeuralNetwork'''


import numpy as np


class DeepNeuralNetwork():
    ''' defines a deep neural network performing
        binary classification:
    '''
    def __init__(self, nx, layers):
        '''constructor'''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list:
            raise TypeError('layers must be a list of positive integers')
        if not all(isinstance(n, int) for n in layers) or\
           all(n <= 0 for n in layers):
            raise TypeError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            wi = 'W'+str(i + 1)
            bi = 'b'+str(i + 1)
            if i == 0:
                self.weights[wi] = np.random.randn(layers[i], layers[i])\
                                   * np.sqrt(2./layers[i-1])
            else:
                self.weights[wi] = np.random.randn(layers[i], layers[i-1])\
                                   * np.sqrt(2/layers[i-1])
            self.weights[bi] = np.zeros((layers[i], 1))
