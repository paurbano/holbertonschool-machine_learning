#!/usr/bin/env python3
'''Scaled Dot Product Attention
https://www.tensorflow.org/tutorials/text/transformer#
scaled_dot_product_attention
'''
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    '''calculates the scaled dot product attention:
    Args:
        Q: is a tensor with its last two dimensions as (..., seq_len_q, dk)
          containing the query matrix
        K: tensor with its last two dimensions as (..., seq_len_v, dk)
            containing the key matrix
        V: tensor with its last two dimensions as (..., seq_len_v, dv)
            containing the value matrix
        mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
            containing the optional mask, or defaulted to None
        if mask is not None, multiply -1e9 to the mask and add it to the
        scaled matrix multiplication
    Returns: output, weights
        outputa: tensor with its last two dimensions as (..., seq_len_q, dv)
                containing the scaled dot product attention
        weights: tensor with its last two dimensions as
                (..., seq_len_q, seq_len_v) containing the attention weights
    '''
    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, V)  # (..., seq_len_q, depth_v)

    return output, attention_weights
