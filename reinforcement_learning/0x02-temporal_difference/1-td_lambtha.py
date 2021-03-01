#!/usr/bin/env python3
'''TD(λ) algorithm'''

import gym
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    '''performs the TD(λ) algorithm
    Args:
        env is the openAI environment instance
        V is a numpy.ndarray of shape (s,) containing the value estimate
        policy: is a function that takes in a state and returns the next
                action to take
        lambtha is the eligibility trace factor
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
    Returns: V, the updated value estimate
    '''
    Et = [0 for i in range(env.observation_space.n)]
    for i in range(episodes):
        state = env.reset()
        for j in range(max_steps):
            # compute the elllgibility
            Et = list(np.array(Et) * lambtha * gamma)
            Et[state] += 1
            # take an action
            action = policy(state)
            new_state, reward, done, info = env.step(action)
            # change the rewards
            if env.desc.reshape(env.observation_space.n)[new_state] == b'H':
                reward = -1
            if env.desc.reshape(env.observation_space.n)[new_state] == b'G':
                reward = 1
            # delta = (R(t+1) * gamma* V(St+1) -V(St))
            delta = reward + gamma * V[new_state] - V[state]
            # V[state] = V[state] + alpha *delta * Et[state]
            V[state] = V[state] + alpha * delta * Et[state]

            if done:
                break
            state = new_state
    return V
