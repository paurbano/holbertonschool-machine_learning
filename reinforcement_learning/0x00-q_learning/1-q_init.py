#!/usr/bin/env python3
'''Initialize Q-table
https://gist.github.com/404akhan/3b0fd788983f17010c761d79a2326a69 - line 12
'''

import numpy as np


def q_init(env):
    '''that initializes the Q-table:
    Args:
        env is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros
    '''
    return np.zeros([env.observation_space.n, env.action_space.n])
