#!/usr/bin/env python3
'''gradient_descent with L2 regularization'''

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''updates the weights and biases of a neural network using
        gradient descent with L2 regularization
    Args:
        @Y: is a one-hot numpy.ndarray of shape (classes, m) that contains
            the correct labels for the data
            classes: is the number of classes
            m: is the number of data points
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs of each layer of the neural network
        alpha: is the learning rate
        lambtha: is the L2 regularization parameter
        L: is the number of layers of the network
    Return
        weights and biases of the network should be updated in place
    '''
    m = len(Y[1])
    # derivative of last Z 'network output'
    dz = cache['A'+str(L)] - Y  # loss, error
    # from that point make the backpropagation
    for i in range(L, 0, -1):
        # Regularitation factor
        L2 = (lambtha / m) * weights['W'+str(i)]
        # derivative of bias bi
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        # derivative of weight wi with regularization
        dW = ((1 / m) * np.matmul(dz, cache['A'+str(i-1)].T)) + L2
        # derivative of Zi
        dz = np.matmul(weights['W'+str(i)].T, dz) *\
            ((1 - cache['A'+str(i-1)] ** 2))

        # update weights and bias
        weights['W'+str(i)] = weights['W'+str(i)] -\
            (alpha * dW)
        weights['b'+str(i)] = weights['b'+str(i)] -\
            (alpha * db)
