#!/usr/bin/env python3
'''Identity Block'''

import tensorflow.keras as K


def identity_block(A_prev, filters):
    '''builds an identity block
    Args:
        A_prev is the output from the previous layer
        filters is a tuple or list containing F11, F3, F12, respectively:
            F11 is the number of filters in the first 1x1 convolution
            F3 is the number of filters in the 3x3 convolution
            F12 is the number of filters in the second 1x1 convolution
        All convolutions inside the block should be followed by batch
        normalization along the channels axis and activation (ReLU).
        All weights should use he normal initialization
        Returns: the activated output of the identity block
    '''
    F11, F3, F12 = filters

    # Save the input value. We'll need this later to add back to the main path.
    X_shortcut = A_prev
    # First component of main path
    A_prev = K.layers.Conv2D(F11, kernel_size=(1, 1), strides=(1, 1),
                             padding='valid',
                             kernel_initializer='he_normal')(A_prev)
    A_prev = K.layers.BatchNormalization(axis=3)(A_prev)
    A_prev = K.layers.Activation('relu')(A_prev)
    # second component of main path
    A_prev = K.layers.Conv2D(F3, kernel_size=(3, 3), strides=(1, 1),
                             padding='same',
                             kernel_initializer='he_normal')(A_prev)
    A_prev = K.layers.BatchNormalization(axis=3)(A_prev)
    A_prev = K.layers.Activation('relu')(A_prev)
    # Third component of main path
    A_prev = K.layers.Conv2D(F12, kernel_size=(1, 1), strides=(1, 1),
                             padding='valid',
                             kernel_initializer='he_normal')(A_prev)
    A_prev = K.layers.BatchNormalization(axis=3)(A_prev)

    # Final step: Add shortcut value to main path,
    # and pass it through a RELU activation
    A_prev = K.layers.Add()([A_prev, X_shortcut])
    A_prev = K.layers.Activation('relu')(A_prev)
    return A_prev
