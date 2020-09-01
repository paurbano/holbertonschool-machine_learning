#!/usr/bin/env python3
'''build neural network with keras'''

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
    model = K.Sequential()
    L2 = K.regularizers.l2(lambtha)
    # Applies Dropout to the input.
    dropout = K.layers.Dropout(1 - keep_prob)
    n = len(layers)
    for i in range(n):
        if i == 0:
            model.add(K.layers.Dense(layers[i], input_shape=(nx,),
                      activation=activations[i], kernel_regularizer='l2'))
        else:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                      kernel_regularizer='l2'))
        if i != n - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
