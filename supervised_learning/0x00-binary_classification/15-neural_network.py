#!/usr/bin/env python3
'''Class NeuralNetwork'''

import numpy as np
import matplotlib.pyplot as plt


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
        cost = self.cost(Y, self.__A2)
        return np.where(self.__A2 >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        '''Calculates one pass of gradient descent on the neural network
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y: array (1,m) correct labels for the input data
            A1: array (1,m) output of the hidden layer
            A2: predicted output
            alpha:  learning rate
        '''
        m = len(A1[0])
        # derivative of Z2
        dz2 = A2 - Y  # loss, error
        # derivative of weights W2
        dw2 = (1 / m) * np.dot(dz2, A1.T)
        # derivative of bias b2
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        # derivative of Z1
        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        # derivative of weights W2
        dW1 = (1 / m) * np.dot(dz1, X.T)
        # derivative of bias b1
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
        # update weights and bias
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        ''' Trains the neural network
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
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.suptitle('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
