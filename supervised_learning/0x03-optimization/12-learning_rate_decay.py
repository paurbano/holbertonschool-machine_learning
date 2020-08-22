#!/usr/bin/env python3
'''tensorflow using inverse time decay'''

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''creates a learning rate decay operation in tensorflow
      using inverse time decay

        @alpha: is the original learning rate
    @decay_rate: weight used to determine the rate at which alpha will decay
        @global_step: number of passes of gradient descent that have elapsed
        @decay_step: number of passes of gradient descent that should occur
                    before alpha is decayed further
        Returns: the learning rate decay operation
    '''
    lrd = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                      decay_rate, staircase=True)
    return lrd
