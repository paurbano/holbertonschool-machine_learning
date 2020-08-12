#!/usr/bin/env python3
'''class DeepNeuralNetwork'''


import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''Calculates one pass of gradient descent on the neural network
            Y: array (1,m) correct labels for the input data
            cache: dictionary with all the intermediary values of the network
            alpha:  learning rate
        '''
        m = len(Y[0])
        # derivative of last Z 'network output'
        dz = cache['A'+str(self.__L)] - Y  # loss, error
        # from that point make the backpropagation
        for i in range(self.__L, 0, -1):
            # derivative of bias bi
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            # derivative of weight wi
            dW = (1 / m) * np.matmul(cache['A'+str(i-1)], dz.T)
            # derivative of Zi
            dz = np.matmul(self.__weights['W'+str(i)].T, dz) *\
                (cache['A'+str(i-1)] * (1 - cache['A'+str(i-1)]))
            # update weights and bias
            self.__weights['W'+str(i)] = self.__weights['W'+str(i)] -\
                (alpha * dW).T
            self.__weights['b'+str(i)] = self.__weights['b'+str(i)] -\
                (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        ''' Trains the neural network
            X: array with shape (nx, m) that contains the input data
                nx: is the number of input features to the neuron
                m: is the number of examples
            Y: array (1,m) with correct labels for the input data
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
            self.gradient_descent(Y, self.__cache, alpha)
        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.suptitle('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
