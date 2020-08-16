#!/usr/bin/env python3
'''accuracy '''

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    ''' calculates the accuracy of a prediction:
        @y: placeholder for the labels of the input data
        @y_pred: tensor containing the network’s predictions
        Returns: a tensor containing the decimal accuracy of the prediction
    '''
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
