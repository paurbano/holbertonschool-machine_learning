#!/usr/bin/env python3
'''class dataset
https://www.tensorflow.org/tutorials/text/transformer
'''

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    '''class dataset'''
    def __init__(self):
        '''constructor
        instance attributes:
        data_train: contains the ted_hrlr_translate/pt_to_en tf.data.Dataset
                    train split, loaded as_supervided
        data_valid: contains the ted_hrlr_translate/pt_to_en tf.data.Dataset
                    validate split, loaded as_supervided
        tokenizer_pt: is the Portuguese tokenizer created from the training set
        tokenizer_en: is the English tokenizer created from the training set
        '''
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

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
