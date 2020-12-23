#!/usr/bin/env python3
'''Unigram BLEU score
https://en.wikipedia.org/wiki/BLEU
https://ariepratama.github.io/Introduction-to-BLEU-in-python/
'''
import numpy as np


def uni_bleu(references, sentence):
    '''calculates the unigram BLEU score for a sentence:
    Args:
        references is a list of reference translations
            each reference translation is a list of the words in
            the translation
        sentence is a list containing the model proposed sentence
    Returns: the unigram BLEU score
    '''
    # score = sentence_bleu(references, sentence)
    # score = corpus_bleu(references, sentence)
    # return score
    c = len(sentence)
    # closest reference length from translation length
    closest_ref_idx = np.argmin([abs(len(x) - c) for x in references])
    r = len(references[closest_ref_idx])
    #
    num_words_in_sentences = {w: sentence.count(w) for w in sentence}
    # refprint(num_words_in_sentences)
    m_max = 0
    for ref in references:
        num_words_in_ref = {w: ref.count(w) for w in ref}
        found = 0
        for key in num_words_in_ref.keys():
            if key in num_words_in_sentences:
                found += 1
        if found > m_max:
            m_max = found
    # precision score
    ps = m_max / c

    # Brevity Penalty
    if c > r:
        BP = 1
    else:
        penality = (1 - float(r) / c)
        BP = np.exp(penality)

    score = BP * ps
    return score
