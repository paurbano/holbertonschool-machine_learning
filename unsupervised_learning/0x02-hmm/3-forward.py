#!/usr/bin/env python3
'''Forward Algorithm HMM'''

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    '''performs the forward algorithm for a hidden markov model
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
    Returns: P, F, or None, None on failure
        P: is the likelihood of the observations given the model
        F: is a numpy.ndarray of shape (N, T) containing the forward path
            probabilities
            F[i, j]: is the probability of being in hidden state i at time j
                    given the previous observations
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
    F = np.zeros([N, T], dtype='float')
    F[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])
    for t in range(1, T):
        F[:, t] = np.multiply(Emission[:, Observation[t]],
                              np.dot(Transition.T, F[:, t - 1]))
    return (np.sum(F[:, T - 1]), F)
