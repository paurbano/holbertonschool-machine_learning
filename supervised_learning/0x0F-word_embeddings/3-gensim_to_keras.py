#!/usr/bin/env python3
'''converts a gensim word2vec model to a keras
'''
from gensim.models import Word2Vec


def gensim_to_keras(model):
    '''converts a gensim word2vec model to a keras
    Args:
        model is a trained gensim word2vec models
    Returns: the trainable keras Embedding
    '''
    return model.wv.get_keras_embedding()
