#!/usr/bin/env python3
'''class DeepNeuralNetwork'''


import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork():
    ''' defines a deep neural network performing
        binary classification:
    '''
    def __init__(self, nx, layers, activation='sig'):
        '''constructor
            nx: number of input features
            layers: list representing number of nodes in each layer
            activation: type of activation function used in the hidden layers
                sig: represents a sigmoid activation
                tanh: represents a tanh activation
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

        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation

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

    @property
    def activation(self):
        '''getter method for activation'''
        return self.__activation

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
            z = np.matmul(self.__weights['W'+str(i)], xi) +\
                self.__weights['b'+str(i)]
            # use softmax for multiclass classification
            # it must calcutate before output layer
            if i == self.__L:
                # softmax as activation function
                t = np.exp(z)
                activ_func = t / np.sum(t, axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    # use sigmoid
                    activ_func = 1 / (1 + np.exp(-z))
                else:
                    # tanh func
                    activ_func = np.sinh(z) / np.cosh(z)

            self.__cache['A'+str(i)] = activ_func
            # self.__cache['A'+str(i)] = softmax
        # return sigmoid, self.__cache
        return activ_func, self.__cache

    def cost(self, Y, A):
        '''Calculates the cost/loss of the model using logistic regression
            Y: is now a one-hot numpy.ndarray of shape (classes, m)
            A: array (1, m) activated output of the neuron for each example
        '''
        m = Y.shape[1]
        '''
        cost = -(1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                 np.multiply(1 - Y, np.log(1.0000001 - A)))
        '''
        # cross - entropy
        cost = -(1 / m) * np.sum(Y * np.log(A))
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
        # return np.where(prediction >= 0.5, 1, 0), cost
        return np.where(prediction == np.max(prediction, axis=0), 1, 0), cost

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
            if self.__activation == 'sig':
                # derivative of Zi
                dz = np.matmul(self.__weights['W'+str(i)].T, dz) *\
                    (cache['A'+str(i-1)] * (1 - cache['A'+str(i-1)]))
            else:
                dz = np.matmul(self.__weights['W'+str(i)].T, dz) *\
                    (1 - cache['A'+str(i-1)] * (cache['A'+str(i-1)]))
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
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose is True or graph is True:
            if type(step) != int:
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        steps = []
        costs = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__cache['A'+str(self.__L)])
            if i % step == 0 or i == iterations:
                costs.append(cost)
                steps.append(i)
                if verbose is True:
                    print('Cost after {} iterations: {}'.format(i, cost))
            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)
        if graph is True:
            plt.plot(np.array(steps), np.array(costs))
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.suptitle('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        ''' Saves the instance object to a file in pickle format
            filename: file to which the object should be saved
        '''
        if '.pkl' not in filename:
            filename = filename + '.pkl'

        # open the file for writing
        with open(filename, 'wb') as fileObject:
            # this writes the object a to the file
            pickle.dump(self, fileObject)

    @staticmethod
    def load(filename):
        ''' Loads a pickled DeepNeuralNetwork object
            filename: is the file from which the object should be loaded
            Returns: the loaded object
        '''
        try:
            with open(filename, 'rb') as fileObject:
                obj = pickle.load(fileObject)
            return obj
        except FileNotFoundError:
            return None
