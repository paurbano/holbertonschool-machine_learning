#!/usr/bin/env python3
''' create layer'''


import tensorflow as tf


def create_layer(prev, n, activation):
    '''create a layer
        prev: tensor output of the previous layer
        n: number of nodes in the layer to create
        activation: activation function that the layer should use
        Returns: the tensor output of the layer
    '''
    kerinit = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    y = tf.layers.dense(prev, activation=activation, units=n, name='layer',
                        kernel_initializer=kerinit, reuse=True)
    return y
