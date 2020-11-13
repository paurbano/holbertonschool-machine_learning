#!/usr/bin/env python3
'''Backward Algorithm
- http://www.adeveloperdiary.com/data-science/machine-learning/
forward-and-backward-algorithm-in-hidden-markov-model/
- https://github.com/zhangyk8/HMM/blob/master/HMM.py
'''

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    '''performs the backward algorithm for a hidden markov model
    Args:
        Observation: is a numpy.ndarray of shape (T,) that contains the index
                    of the observation
            T is the number of observations
        Emission: is a numpy.ndarray of shape (N, M) containing the emission
                probability of a specific observation given a hidden state
            Emission[i, j] is the probability of observing j given the hidden
                           state i
            N is the number of hidden states
            M is the number of all possible observations
        Transition: is a 2D numpy.ndarray of shape (N, N) containing the
                    transition probabilities
            Transition[i, j] is the probability of transitioning from the
                            hidden state i to j
        Initial: a numpy.ndarray of shape (N, 1) containing the probability of
                starting in a particular hidden state
    Returns: P, B, or None, None on failure
        P is the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the backward path
            probabilities
            B[i, j] is the probability of generating the future observations
                from hidden state i at time j
    '''
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return (None, None)
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2 or \
       Transition.shape[0] != Transition.shape[1]:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    if Emission.shape[0] != Transition.shape[0] != Transition.shape[0] !=\
       Initial.shape[0]:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    B = np.empty([N, T], dtype='float')
    B[:, T - 1] = 1
    for t in reversed(range(T - 1)):
        B[:, t] = np.dot(Transition,
                         np.multiply(Emission[:,
                                     Observation[t + 1]], B[:, t + 1]))
    P = np.dot(Initial.T, np.multiply(Emission[:, Observation[0]], B[:, 0]))
    return (P, B)
