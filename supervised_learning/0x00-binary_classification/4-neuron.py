#!/usr/bin/env python3
''' My first Neuron'''

import numpy as np


class Neuron():
    '''class Neuron'''
    def __init__(self, nx):
        '''constructor'''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''getter for Weigth'''
        return self.__W

    @property
    def b(self):
        '''getter for bias'''
        return self.__b

    @property
    def A(self):
        '''getter for activation function'''
        return self.__A

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neuron with sigmoid'''
        x = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp(-x))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression'''
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                 np.multiply(1 - Y, np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the neuron’s predictions'''
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        return np.where(self.__A >= 0.5, 1, 0), cost
