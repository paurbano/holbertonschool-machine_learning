#!/usr/bin/env python3

uni_bleu = __import__('0-uni_bleu').uni_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]
#references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
#sentence = [['this', 'is', 'a', 'test']]
print(uni_bleu(references, sentence))
