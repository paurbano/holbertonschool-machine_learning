#!/usr/bin/env python3
'''build neural network with keras Functional API'''


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''builds a neural network with the Keras library:
    Args:
        nx: is the number of input features to the network
        layers: is a list containing the number of nodes in each
                layer of the network
        activations: is a list containing the activation functions
                    used for each layer of the network
        lambtha: is the L2 regularization parameter
        keep_prob: is the probability that a node will be kept for dropout

    Returns: the keras model
    '''
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    dense = K.layers.Dense(layers[0], activation=activations[0],
                           kernel_regularizer=L2)
    x = dense(inputs)
    x = K.layers.Dropout(1 - keep_prob)(x)
    n = len(layers)
    for i in range(1, n):
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=L2)(x)
        if i < n - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    '''outputs = K.layers.Dense(layers[i + 1], activation=activations[i + 1],
                             kernel_regularizer=L2)(x)'''
    model = K.Model(inputs=inputs, outputs=x)
    return model
