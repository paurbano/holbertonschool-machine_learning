#!/usr/bin/env python3
'''Create Masks
https://www.tensorflow.org/tutorials/text/transformer#masking
'''
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    '''creates all masks for training/validation
    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in) that contains the
                input sentence
        target: tf.Tensor of shape (batch_size, seq_len_out) that contains the
                target sentence
    Returns: encoder_mask, look_ahead_mask, decoder_mask
        encoder_mask: tf.Tensor padding mask of shape
                    (batch_size, 1, 1,seq_len_in) to be applied in the encoder
        look_ahead_mask: tf.Tensor look ahead mask of shape
                        (batch_size, 1, seq_len_out, seq_len_out) to be
                        applied in the decoder
        decoder_mask: tf.Tensor padding mask of shape
                    (batch_size, 1, 1,seq_len_in) to be applied in the decoder
    '''
    # add extra dimensions to add the padding
    # to the attention logits.
    # (batch_size, 1, 1, seq_len_in)
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]
    # (batch_size, 1, 1, seq_len_in)
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    size = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    dec_target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_mask = dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return encoder_mask, combined_mask, decoder_mask
