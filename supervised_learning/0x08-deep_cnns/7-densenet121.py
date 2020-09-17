#!/usr/bin/env python3
'''DenseNet-121'''

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    '''builds the DenseNet-121 architecture
    Args:
        growth_rate is the growth rate
        compression is the compression factor
        All convolutions should be preceded by Batch Normalization and
        a rectified linear activation (ReLU), respectively
    Returns: the keras model
    '''
    # input
    input_data = K.Input(shape=(224, 224, 3))

    # From architecture for ImageNet
    nb_filter = 64
    nb_layers = [6, 12, 24, 16]  # For DenseNet-121

    # Initial convolution
    x = K.layers.BatchNormalization(axis=3)(input_data)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(nb_filter, kernel_size=(7, 7), strides=(2, 2),
                        kernel_initializer='he_normal', padding='same')(x)
    # x = K.layers.ZeroPadding2D((1, 1))(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add dense blocks
    # in DenseNet-121 is 4, you could add a parameter for this!!
    for idx in range(len(nb_layers) - 1):
        x, nb_filter = dense_block(x, nb_filter, growth_rate, nb_layers[idx])
        # Add transition_block
        x, nb_filter = transition_layer(x, nb_filter, compression)

    x, nb_filter = dense_block(x, nb_filter, growth_rate, nb_layers[idx + 1])
    x = K.layers.AveragePooling2D((7, 7))(x)
    x = K.layers.Dense(1000, activation='softmax')(x)

    model = K.Model(inputs=input_data, outputs=x)

    return model
