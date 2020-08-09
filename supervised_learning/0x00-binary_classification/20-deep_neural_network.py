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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError('layers must be a list of positive integers')

            wi = 'W'+str(i + 1)
            bi = 'b'+str(i + 1)
            if i == 0:
                self.__weights[wi] = np.random.randn(layers[i], nx)\
                                   * np.sqrt(2./nx)
            else:
                self.__weights[wi] = np.random.randn(layers[i], layers[i-1])\
                                   * np.sqrt(2/layers[i-1])
            self.__weights[bi] = np.zeros((layers[i], 1))

    @property
    def L(self):
        '''getter method for number of layers'''
        return self.__L

    @property
    def cache(self):
        '''getter method for intermediate values'''
        return self.__cache

    @property
    def weights(self):
        '''getter method for weights and bias'''
        return self.__weights

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neural network
            X: array with shape (nx, m) that contains the input data
            nx: is the number of input features to the neuron
            m: is the number of examples
            Returns: output of the neural network and the cache, respectively
        '''
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            '''
            # get previous input/activation data, something like this
            z1 = np.dot(W1, X) + b1
            a1 = sigmoid(z1)
            z2 = np.dot(W2, a1) + b2
            a2 = sigmoid(z2)
            z3 = np.dot(W3, a2) + b3
            a3 = sigmoid(z3)
             ....
            zi = np.dot(Wi, ai-1) + bi
            ai = sigmoid(zi)
            '''
            xi = self.__cache['A'+str(i-1)]
            z = np.dot(self.__weights['W'+str(i)], xi) +\
                self.__weights['b'+str(i)]

            sigmoid = 1 / (1 + np.exp(-z))
            self.__cache['A'+str(i)] = sigmoid

        return sigmoid, self.__cache

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression
            Y: array (1, m) contains the correct labels for the input data
            A: array (1, m) activated output of the neuron for each example
        '''
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                 np.multiply(1 - Y, np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the neural network’s predictions
            X: array (nx,m) contains the input data
                nx:  number of input features to the neuron
                m: number of examples
            Y: array (1, m) contains the correct labels for the input data
            return:neuron’s prediction and the cost of the network
        '''
        prediction, cache = self.forward_prop(X)
        cost = self.cost(Y, prediction)
        return np.where(prediction >= 0.5, 1, 0), cost
