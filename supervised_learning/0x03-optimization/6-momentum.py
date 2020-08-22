#!/usr/bin/env python3
'''using the gradient descent with momentum optimization algorithm'''


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    ''' training operation using the gradient descent
        with momentum optimization algorithm

        @loss: is the loss of the network
        @alpha: is the learning rate
        @beta1: is the momentum weight
        Returns: the momentum optimization operation
    '''
    momentum = tf.train.MomentumOptimizer(alpha, beta1)
    compute_gradients = momentum.compute_gradients(loss)
    apply_gradients = momentum.apply_gradients(compute_gradients)
    return apply_gradients
