#!/usr/bin/env python3
'''Forward Propagation with Dropout'''


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    '''forward propagation using Dropout:
    Args:
        X is a numpy.ndarray of shape (nx, m) containing the input data for
            the network
            nx is the number of input features
            m is the number of data points
        weights: dictionary of the weights and biases of the neural network
        L: the number of layers in the network
        keep_prob: is the probability that a node will be kept
    Returns:
        a dictionary containing the outputs of each layer and the dropout mask
        used on each layer
    '''
    output_drop = {}
    A = X
    for i in range(L):
        if i != 0:
            # generate dropout mask
            output_drop['D'+str(i)] = (np.random.rand(A.shape[0], A.shape[1])
                                       < keep_prob).astype(int)
            A = np.multiply(A, output_drop['D'+str(i)])
            A = A / keep_prob

        output_drop['A'+str(i)] = A
        z = np.dot(weights['W' + str(i+1)], A) + weights['b' + str(i+1)]
        # use softmax for multiclass classification
        # it must calcutate before output layer
        if i == L - 1:
            # softmax as activation function
            t = np.exp(z)
            A = t / np.sum(t, axis=0, keepdims=True)
            # add output of layer to dictionary
            output_drop['A'+str(i+1)] = A
        else:
            # use tanh for hidden layers
            A = np.sinh(z) / np.cosh(z)
    return output_drop
