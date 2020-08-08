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

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neural network
            X: array (nx,m) contains the input data
                nx:  number of input features to the neuron
                m: number of examples
        '''
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression
            Y: array (1, m) contains the correct labels for the input data
            A: array (1, m) containing the activated output of the neuron
               for each example
            Note : 1.0000001 - A instead of 1 - A to avoid division by 0
        '''
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                 np.multiply(1 - Y, np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the neuron’s predictions
            X: array (nx,m) contains the input data
                nx:  number of input features to the neuron
                m: number of examples
            Y: array (1, m) contains the correct labels for the input data
            return:neuron’s prediction and the cost of the network
        '''
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        return np.where(self.__A >= 0.5, 1, 0), cost
