#!/usr/bin/env python3
'''use of placeholders'''

import tensorflow as tf


def create_placeholders(nx, classes):
    '''returns two placeholders, x and y
        nx: the number of feature columns in our data
        classes: the number of classes in our classifier
        return: placeholders named x and y
            x is the placeholder for the input data
            y is the placeholder for the one-hot labels
    '''
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
