#!/usr/bin/env python3
'''inception block'''

import tensorflow.keras as K


def inception_block(A_prev, filters):
    '''builds an inception block
    Args:
        A_prev is the output from the previous layer
        filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
        respectively:
            F1 is the number of filters in the 1x1 convolution
            F3R is the number of filters in the 1x1 convolution before
            the 3x3 convolution
            F3 is the number of filters in the 3x3 convolution
            F5R is the number of filters in the 1x1 convolution before
            the 5x5 convolution
            F5 is the number of filters in the 5x5 convolution
            FPP is the number of filters in the 1x1 convolution after
            the max pooling
        All convolutions inside the inception block should use a rectified
        linear activation (ReLU)
    Returns: the concatenated output of the inception block
    '''
    # filters
    F1 = filters[0]
    F3R = filters[1]
    F3 = filters[2]
    F5R = filters[3]
    F5 = filters[4]
    FPP = filters[5]
    # 1st layer
    conv_1x1 = K.layers.Conv2D(F1, (1, 1),
                               padding='same', activation='relu')(A_prev)

    conv_3x3 = K.layers.Conv2D(F3R, (1, 1),
                               padding='same', activation='relu')(A_prev)
    conv_3x3 = K.layers.Conv2D(F3, (3, 3),
                               padding='same', activation='relu')(conv_3x3)

    conv_5x5 = K.layers.Conv2D(F5R, (1, 1),
                               padding='same', activation='relu')(A_prev)
    conv_5x5 = K.layers.Conv2D(F5, (5, 5),
                               padding='same', activation='relu')(conv_5x5)

    pool_proj = K.layers.MaxPool2D((3, 3),
                                   strides=(1, 1), padding='same')(A_prev)
    pool_proj = K.layers.Conv2D(FPP, (1, 1),
                                padding='same', activation='relu')(pool_proj)

    output = K.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj],
                                  axis=3)

    return output
