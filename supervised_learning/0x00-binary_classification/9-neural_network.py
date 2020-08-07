#!/usr/bin/env python3
'''Class NeuralNetwork'''

import numpy as np


class NeuralNetwork():
    '''defines a neural network with one
       hidden layer performing binary classification

       Attributes
        nx: is the number of input features
        nodes: number of nodes in the hidden layer
        W1: weights vector for the hidden layer
        b1: bias for the hidden layer
        A1: activated output for the hidden layer
        W2: weights vector for the output neuron
        b2: bias for the output neuron.
        A2: activated output for the output neuron (prediction)
    '''
    def __init__(self, nx, nodes):
        '''constructor'''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''getter method for W1'''
        return self.__W1

    @property
    def b1(self):
        '''getter for b1'''
        return self.__b1

    @property
    def A1(self):
        '''getter for A1'''
        return self.__A1

    @property
    def W2(self):
        '''getter method for W2'''
        return self.__W2

    @property
    def b2(self):
        '''getter for b2'''
        return self.__b2

    @property
    def A2(self):
        '''getter for A2'''
        return self.__A2
