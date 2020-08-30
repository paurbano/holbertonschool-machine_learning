#!/usr/bin/env python3
'''gradient_descent with L2 regularization'''

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    '''updates the weights of a neural network with Dropout
        regularization using gradient descen
    Args:
        @Y: is a one-hot numpy.ndarray of shape (classes, m) that contains
            the correct labels for the data
            classes: is the number of classes
            m: is the number of data points
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs and dropout masks of each layer of
                the neural network
        alpha: is the learning rate
        keep_prob is the probability that a node will be kept
        L: is the number of layers of the network
    Return
        weights and biases of the network should be updated in place
    '''
    m = len(Y[1])
    # derivative of last Z 'network output'
    dz = cache['A'+str(L)] - Y  # loss, error
    # from that point make the backpropagation
    for i in range(L, 0, -1):
        # derivative of bias bi
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        # derivative of weight wi
        dW = ((1 / m) * np.matmul(dz, cache['A'+str(i-1)].T))
        # derivative of Zi with dropout
        if i - 1 > 0:
            dz = np.matmul(weights['W'+str(i)].T, dz) *\
                 ((1 - cache['A'+str(i-1)] ** 2)) *\
                 (cache['D'+str(i-1)] / keep_prob)

        # update weights and bias
        weights['W'+str(i)] = weights['W'+str(i)] -\
            (alpha * dW)
        weights['b'+str(i)] = weights['b'+str(i)] -\
            (alpha * db)
