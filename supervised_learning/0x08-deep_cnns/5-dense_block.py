#!/usr/bin/env python3
'''Dense block DenseNet'''

import tensorflow.keras as K


def conv_block(x, nb_filter):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D
        # Arguments
            x: input tensor
            nb_filter: number of filters
        Return:  Convolution Block
    '''

    # 1x1 Convolution
    inter_channel = nb_filter * 4
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)
    # (Bottleneck layer)
    x = K.layers.Convolution2D(inter_channel, kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer='he_normal')(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)
    # 3x3 Convolution
    x = K.layers.Convolution2D(nb_filter, kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer='he_normal')(x)

    return x


def dense_block(X, nb_filters, growth_rate, layers):
    '''builds a dense block
    Args:
        X is the output from the previous layer
        nb_filters is an integer representing the number of filters in X
        growth_rate is the growth rate for the dense block
        layers is the number of layers in the dense block
        All weights should use he normal initialization
        All convolutions should be preceded by Batch Normalization and a
            rectified linear activation (ReLU), respectively
        Returns: The concatenated output of each layer within the Dense Block
        and the number of filters within the concatenated outputs, respectively
    '''
    concat_feat = X
    for i in range(layers):
        X = conv_block(concat_feat, growth_rate)
        concat_feat = K.layers.concatenate([concat_feat, X], axis=3)

        nb_filters += growth_rate

    return concat_feat, nb_filters
