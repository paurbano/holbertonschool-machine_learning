#!/usr/bin/env python3
'''class SelfAttention
https://arxiv.org/pdf/1409.0473.pdf - pag 13
'''
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    '''class SelfAttention'''
    def __init__(self, units):
        '''Constructor
        Args:
        units: is an integer representing the number of hidden units in the
               alignment model
        Sets the following public instance attributes:
            W - a Dense layer with units units, to be applied to the previous
                decoder hidden state
            U - a Dense layer with units units, to be applied to the encoder
                hidden states
            V - a Dense layer with 1 units, to be applied to the tanh of the
                sum of the outputs of W and U
        '''
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        '''call method
        Args:
            s_prev: is a tensor of shape (batch, units) containing the previous
                    decoder hidden state
            hidden_states: is a tensor of shape (batch, input_seq_len, units)
                            containing the outputs of the encoder
        Returns: context, weights
            context: is a tensor of shape (batch, units) that contains the
                    context vector for the decoder
            weights: is a tensor of shape (batch, input_seq_len, 1) that
                    contains the attention weights
        '''
        s_prev = tf.expand_dims(s_prev, 1)
        e = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        a = tf.nn.softmax(e, axis=1)
        c = a * hidden_states
        c = tf.reduce_sum(c, axis=1)
        return c, a
