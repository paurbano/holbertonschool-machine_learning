#!/usr/bin/env python3
'''batch normalization layer for a neural network'''


import tensorflow as tf
import numpy as np


def create_batch_norm_layer(prev, n, activation):
    '''creates a batch normalization layer for a neural network in tensorflow

        @prev: is the activated output of the previous layer
        @n: is the number of nodes in the layer to be created
        @activation: is the activation function that should be used
                     on the output of the layer
        Returns: a tensor of the activated output for the layer
    '''
    # create layer
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    # in this case activation func apply when return for output
    y = tf.layers.Dense(units=n, kernel_initializer=init, name='layer')
    # make prediction
    x = y(prev)
    # calculates variables/tendors for batch normalization
    mean, variance = tf.nn.moments(x, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    epsilon = 1e-8
    # normalize tensor with batch normalization method
    norma = tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)
    # now apply activation function for output
    return activation(norma)
