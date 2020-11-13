#!/usr/bin/env python3
'''Viretbi Algorithm
- https://www.adeveloperdiary.com/data-science/machine-learning/
implement-viterbi-algorithm-in-hidden-markov-model-using-python-and-r/
- http://www.blackarbs.com/blog/
introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
- https://github.com/zhangyk8/HMM/blob/master/HMM.py
'''

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    '''calculates the most likely sequence of hidden states for a
       hidden markov mode
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
    Returns: path, P, or None, None on failure
        path is the a list of length T containing the most likely sequence of
             hidden states
        P is the probability of obtaining the path sequence
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
    # delta --> highest probability of any path that reaches state i
    d = np.zeros([N, T])
    # phi --> argmax by time step for each state
    f = np.empty([N, T], dtype=int)
    # # init delta and phi 
    d[:, 0] = np.multiply(Initial.T, Emission[:, Observation[0]])
    # Recursive case
    for t in range(1, T):
        for i in range(N):
            d[i, t] = np.max(d[:, t - 1] * Transition[:, i]) *\
                      Emission[i, Observation[t]]
            f[i, t] = np.argmax(d[:, t - 1] * Transition[:, i])

    # Path Array
    path = np.zeros(T)

    # Find the most probable last hidden state
    # x = np.argmax(d[:,N-1])
    # path= np.argmax(d[:,N-1]))

    path[T - 1] = np.argmax(d[:, T - 1])
    # backtrack_index = 1
    for i in range(T - 2, -1, -1):
        path[i] = f[int(path[i + 1]), i + 1]
        # backtrack_index += 1
    P = np.max(d[:, T - 1:], axis=0)[0]
    # Flip the path array since we were backtracking
    # path = np.flip(path, axis=0)
    path = [int(i) for i in path]
    return (path, P)
