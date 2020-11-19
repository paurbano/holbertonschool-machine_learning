#!/usr/bin/env python3
'''Absorbing Chains '''

import numpy as np


def absorbing(P):
    '''determines if a markov chain is absorbing:
    Args:
        P: is a is a square 2D numpy.ndarray of shape (n, n) representing the
            standard transition matrix
           P[i, j] is the probability of transitioning from state i to state j
           n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    '''
    if type(P) is not np.ndarray or len(P.shape) != 2 or\
       P.shape[0] != P.shape[1]:
        return None
    if np.any(P < 0):
        return None

    if np.min(P ** 2) < 0 or np.min(P ** 3) < 0:
        return None

    P = P.copy()
    absorb = np.ndarray(P.shape[0])
    while True:
        prev = absorb.copy()
        absorb = np.any(P == 1, axis=0)
        if absorb.all():
            return True
        if np.all(absorb == prev):
            return False
        absorbed = np.any(P[:, absorb], axis=1)
        P[absorbed, absorbed] = 1
