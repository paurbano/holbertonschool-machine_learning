#!/usr/bin/env python3
''' My first Neuron'''

import numpy as np
import matplotlib.pyplot as plt


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
        '''Evaluates the neuron’s predictions
        X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
        Y: array (1,m) correct labels for the input data
        return: neuron’s prediction and the cost of the network
        '''
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        return np.where(self.__A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''Calculates one pass of gradient descent on the neuron
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y: array (1,m) correct labels for the input data
            A: array (1,m) with activated output
            alpha:  learning rate
        '''
        m = len(X[0])
        dz = A - Y
        dzT = dz.transpose()
        dw = (1 / m) * (np.matmul(X, dzT))
        db = (1 / m) * np.sum(dz)
        dwT = (alpha * dw).T
        self.__W = self.__W - dwT
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        ''' Trains the neuron
            X: numpy.ndarray with shape (nx, m) that contains the input data
                    nx is the number of input features to the neuron
                    m is the number of examples
                Y: array (1,m) correct labels for the input data
                A: array (1,m) with activated output
                alpha: learning rate
                iterations: number of iterations to train over
        '''
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        steps = list(range(0, iterations + 1, step))
        costs = []
        msg = "Cost after {iteration} iterations: {cost}"
        for iter in range(iterations + 1):
            if verbose and iter in steps:
                p, c = self.evaluate(X, Y)
                costs.append(c)
                print(msg.format(iteration=iter, cost=c))
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.suptitle('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
