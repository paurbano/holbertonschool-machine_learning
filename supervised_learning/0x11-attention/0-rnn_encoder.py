#!/usr/bin/env python3
'''class RNNEncoder
https://towardsdatascience.com/sequence-to-sequence-models-from-rnn-to-
transformers-e24097069639
'''
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    '''class RNNEncoder'''
    def __init__(self, vocab, embedding, units, batch):
        '''constructor
        Args:
            vocab: integer representing the size of the input vocabulary
            embedding: integer representing the dimensionality of the
                        embedding vector
            units: integer representing the number of hidden units in the
                    RNN cell
            batch: integer representing the batch size
        '''
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(self.units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        '''Initializes the hidden states for the RNN cell to a tensor of zeros
            Returns: a tensor of shape (batch, units) containing the
                    initialized hidden states
        '''
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        '''
        Args:
            x: is a tensor of shape (batch, input_seq_len) containing the input
              to the encoder layer as word indices within the vocabulary
            initial: is a tensor of shape (batch, units) containing the initial
                    hidden state
        Returns: outputs, hidden
            outputs: tensor of shape (batch, input_seq_len, units)containing
                    the outputs of the encoder
            hidden: tensor of shape (batch, units) containing the last
                    hidden state of the encoder
        '''
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
