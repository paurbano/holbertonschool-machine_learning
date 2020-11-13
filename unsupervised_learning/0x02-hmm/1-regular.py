#!/usr/bin/env python3
'''Regular Chains
https://math.libretexts.org/Bookshelves/Applied_Mathematics/Book%
3A_Applied_Finite_Mathematics_(Sekhon_and_Bloom)/10%3A_Markov_Chains/
10.03%3A_Regular_Markov_Chains
'''

import numpy as np


def regular(P):
    '''determines the steady state probabilities of a regular markov chain
    Args:
        P: is a is a square 2D numpy.ndarray of shape (n, n) representing the
            transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state
            probabilities, or None on failure
    '''
    if type(P) is not np.ndarray or len(P.shape) != 2 or\
       P.shape[0] != P.shape[1]:
        return None
    m = ((len(P.shape) - 1) ** 2) + 1
    T = P.copy()
    for i in range(m):
        T = np.linalg.matrix_power(T, 2)
        if np.any(T <= 0):
            return None
    for i in range(10):
        P = np.linalg.matrix_power(P, 2)

    return np.array([P[0]])
