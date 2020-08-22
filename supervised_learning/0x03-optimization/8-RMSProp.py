#!/usr/bin/env python3
'''training operation for a neural network in tensorflow using the RMSProp'''

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    '''training operation using the RMSProp optimization algorithm
    @loss: is the loss of the network
    @alpha: is the learning rate
    @beta2: is the RMSProp weight
    @epsilon: is a small number to avoid division by zero
    Returns: the RMSProp optimization operation
    '''
    rms = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    compute_gradients = rms.compute_gradients(loss)
    apply_gradients = rms.apply_gradients(compute_gradients)
    return apply_gradients
