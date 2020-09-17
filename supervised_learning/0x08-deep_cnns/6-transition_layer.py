#!/usr/bin/env python3
'''transition layer'''

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    '''builds a transition layer
    Args:
        X is the output from the previous layer
        nb_filters is an integer representing the number of filters in X
        compression is the compression factor for the transition layer
    Returns: The output of the transition layer and the number of filters
            within the output, respectively
    '''
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(int(nb_filters * compression), kernel_size=(1, 1),
                        kernel_initializer='he_normal', padding='same')(X)
    X = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(X)

    return X, int(nb_filters * compression)
