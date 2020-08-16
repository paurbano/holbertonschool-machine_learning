#!/usr/bin/env python3
'''Forward Propagation'''


import tensorflow as tf


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    '''creates the forward propagation graph for the neural network
        x: placeholder for the input data
        layer_sizes: list with the number of nodes in each layer
                     of the network
        activations: list with the activation functions for each layer
                     of the network
        Returns: the prediction of the network in tensor form
    '''
    # if number of layers are equal to activation functions
    if len(layer_sizes) == len(activations):
        # prediction for first layer
        Z = create_layer(x, layer_sizes[0], activations[0])
        # that is the input for next and so on...
        for i in range(1, len(layer_sizes)):
            Z = create_layer(Z, layer_sizes[i], activations[i])
        return Z
    return None
