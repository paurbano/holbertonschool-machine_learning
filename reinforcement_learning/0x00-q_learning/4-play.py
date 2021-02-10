#!/usr/bin/env python3
'''Play
https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/
master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb
'''
import gym
import numpy as np


def play(env, Q, max_steps=100):
    '''has the trained agent play an episode:
    Args:
        env is the FrozenLakeEnv instance
        Q is a numpy.ndarray containing the Q-table
        max_steps is the maximum number of steps in the episode
    Each state of the board should be displayed via the console
    You should always exploit the Q-table
    Returns: the total rewards for the episode
    '''
    state = env.reset()
    env.render()
    for step in range(max_steps):
        # Take the action (index) that have the maximum expected future reward
        # given that state
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        # Here, print the state (to see if our agent is
        # on the goal or fall into an hole)
        env.render()

        if done:
            break
        state = new_state
    return reward
