#!/usr/bin/env python3
'''Epsilon Greedy
https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/
master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb
'''

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    '''uses epsilon-greedy to determine the next action:
    Args:
        Q is a numpy.ndarray containing the q-table
        state is the current state
        epsilon is the epsilon to use for the calculation
    Returns: the next action index
    '''
    # First we randomize a number
    p = np.random.uniform(0, 1)

    # If this number > greater than epsilon -->
    # exploitation (taking the biggest Q value for this state)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    # Else doing a random choice --> exploration
    else:
        # action = env.action_space.sample()
        action = np.random.randint(0, int(Q.shape[1]))
    return action
