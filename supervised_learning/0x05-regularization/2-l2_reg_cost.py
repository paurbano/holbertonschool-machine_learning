#!/usr/bin/env python3
''' L2 regularization'''

import tensorflow as tf


def l2_reg_cost(cost):
    '''L2 regularization'''
    l2 = cost + tf.losses.get_regularization_losses()
    return l2
