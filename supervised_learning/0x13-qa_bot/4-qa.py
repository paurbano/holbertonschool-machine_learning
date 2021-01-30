#!/usr/bin/env python3
'''Multi-reference Question Answering'''

question_answer = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def qa_bot(corpus_path):
    '''answers questions from multiple reference texts
    Args:
        corpus_path is the path to the corpus of reference documents
    '''
    while(True):
        question = input('Q: ')
        question = question.lower()
        if question == 'exit' or question == 'quit' or\
                question == 'goodbye' or question == 'bye':
            print('A: Goodbye')
            break
        else:
            reference = semantic_search(corpus_path, question)
            answer = question_answer(question, reference)
            if answer is None:
                answer = 'Sorry, I do not understand your question.'
            print('A: {}'.format(answer))
