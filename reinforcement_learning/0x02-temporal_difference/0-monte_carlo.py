#!/usr/bin/env python3
'''Monte Carlo'''

import numpy as np
import gym


def generate_episode(env, policy, max_steps):
    '''Generate episode
    '''
    episode = [[], []]
    state = env.reset()
    for t in range(max_steps):
        action = policy(state)
        new_state, reward, done, info = env.step(action)
        episode[0].append(state)
        if env.desc.reshape(env.observation_space.n)[new_state] == b'H':
            episode[1].append(-1)
            return episode
        if env.desc.reshape(env.observation_space.n)[new_state] == b'G':
            episode[1].append(1)
            return episode
        episode[1].append(reward)
        state = new_state
    return episode


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    '''performs the Monte Carlo algorithm
    Args:
        env: is the openAI environment instance
        V: is a numpy.ndarray of shape (s,) containing the value estimate
        policy: is a function that takes in a state and returns the next
                action to take
        episodes: is the total number of episodes to train over
        max_steps: is the maximum number of steps per episode
        alpha: is the learning rate
        gamma: is the discount rate
    Returns: V, the updated value estimate
    '''
    discounts = [gamma ** i for i in range(max_steps)]
    for ep in range(episodes):
        episode = generate_episode(env, policy, max_steps)
        for i in range(len(episode[0])):
            Gt = sum(np.array(episode[1][i:]) *
                     np.array(discounts[:len(episode[1][i:])]))
            # V(St) = V(St) + alplha * (Gt - V(St))
            V[episode[0][i]] = V[episode[0][i]] +\
                alpha * (Gt - V[episode[0][i]])
    return V
