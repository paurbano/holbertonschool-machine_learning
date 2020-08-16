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
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    '''y = tf.layers.dense(prev, units=n, activation=activation,
            kernel_initializer=kerinit, name='layer')'''
    y = tf.layers.Dense(n, activation, kernel_initializer=init, name='layer')
    return y(prev)
