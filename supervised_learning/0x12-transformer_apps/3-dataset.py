#!/usr/bin/env python3
'''class dataset
https://www.tensorflow.org/tutorials/text/transformer#positional_encoding
https://www.tensorflow.org/datasets/keras_example
'''

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    '''class dataset'''
    def __init__(self, batch_size, max_len):
        '''constructor
        instance attributes:
        data_train: contains the ted_hrlr_translate/pt_to_en tf.data.Dataset
                    train split, loaded as_supervided
        data_valid: contains the ted_hrlr_translate/pt_to_en tf.data.Dataset
                    validate split, loaded as_supervided
        tokenizer_pt: is the Portuguese tokenizer created from the training set
        tokenizer_en: is the English tokenizer created from the training set
        '''
        self.MAX_LENGTH = max_len
        data_train, data_info = tfds.load('ted_hrlr_translate/pt_to_en',
                                          split='train', as_supervised=True,
                                          with_info=True)
        data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='validation', as_supervised=True)
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        data_train = data_train.map(self.tf_encode)
        data_train = data_train.filter(self.filter_max_length)
        data_train = data_train.cache()
        data_train = data_train.shuffle(data_info.splits['train'].num_examples)
        self.data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)

        data_valid = data_valid.map(self.tf_encode)
        data_valid = data_valid.filter(self.filter_max_length).\
            padded_batch(batch_size)
        self.data_valid = data_valid

    def tokenize_dataset(self, data):
        '''creates sub-word tokenizers for our dataset
        Args:
        data: tf.data.Dataset whose examples are formatted as a tuple (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English sentence
        Returns: tokenizer_pt, tokenizer_en
        tokenizer_pt is the Portuguese tokenizer
        tokenizer_en is the English tokenizer
        '''
        token_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                    (en.numpy() for pt, en in data), target_vocab_size=2**15)

        token_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                    (pt.numpy() for pt, en in data), target_vocab_size=2**15)

        return token_pt, token_en

    def encode(self, pt, en):
        '''encodes a translation into tokens
        Args:
        pt is the tf.Tensor containing the Portuguese sentence
        en is the tf.Tensor containing the corresponding English sentence
        Returns: pt_tokens, en_tokens
        pt_tokens is a np.ndarray containing the Portuguese tokens
        en_tokens is a np.ndarray. containing the English tokens
        '''
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
                    pt.numpy()) + [self.tokenizer_pt.vocab_size+1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size+1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        '''acts as a tensorflow wrapper for the encode instance method
        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence
        '''
        result_pt, result_en = tf.py_function(encode, [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

    def filter_max_length(self, x, y):
        '''filter method'''
        max_length = self.MAX_LENGTH
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)
