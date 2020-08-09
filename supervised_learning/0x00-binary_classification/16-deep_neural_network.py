#!/usr/bin/env python3
'''class DeepNeuralNetwork'''


import numpy as np


class DeepNeuralNetwork():
    ''' defines a deep neural network performing
        binary classification:
    '''
    def __init__(self, nx, layers):
        '''constructor
            nx: number of input features
            layers: list representing number of nodes in each layer
            L: The number of layers
            cache: dictionary to hold all intermediary values of the network
            weights: dictionary to hold all weights and biased of the network
        '''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or not layers:
            raise TypeError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError('layers must be a list of positive integers')

            wi = 'W'+str(i + 1)
            bi = 'b'+str(i + 1)
            if i == 0:
                self.weights[wi] = np.random.randn(layers[i], nx)\
                                   * np.sqrt(2./nx)
            else:
                self.weights[wi] = np.random.randn(layers[i], layers[i-1])\
                                   * np.sqrt(2/layers[i-1])
            self.weights[bi] = np.zeros((layers[i], 1))
