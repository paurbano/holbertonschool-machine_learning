#!/usr/bin/env python3
'''inception network
source: https://www.analyticsvidhya.com/blog/2018/10/
        understanding-inception-network-from-scratch/
'''

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    '''builds the inception network
        assume the input data will have shape (224, 224, 3)
        All convolutions inside and outside the inception block should use
        activation (ReLU)
        Returns: the keras model
    '''
    input_layer = K.Input(shape=(224, 224, 3))
    x = K.layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2),
                        activation='relu')(input_layer)
    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
    x = K.layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1),
                        activation='relu')(x)
    x = K.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1),
                        activation='relu')(x)
    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
    # inception 3a, 3b
    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    # max pool
    x = K.layers.MaxPool2D((3, 3), padding="same", strides=(2, 2))(x)
    # inception 4a, 4b, 4c, 4d, 4e
    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    # max pool
    x = K.layers.MaxPool2D((3, 3), padding="same", strides=(2, 2))(x)
    # inception 5a, 5b
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])
    # avg pool
    x = K.layers.AveragePooling2D((7, 7), strides=1)(x)
    # dropout
    x = K.layers.Dropout(0.4)(x)
    # linear
    # softmax
    x = K.layers.Dense(1000, activation='softmax')(x)
    model = K.Model(inputs=input_layer, outputs=x)
    return model
