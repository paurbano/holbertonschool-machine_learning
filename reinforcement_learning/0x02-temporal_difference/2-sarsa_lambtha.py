#!/usr/bin/env python3
'''SARSA(λ)'''
import gym
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


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    '''performs SARSA(λ)
    Args:
        env is the openAI environment instance
        Q is a numpy.ndarray of shape (s,a) containing the Q table
        lambtha is the eligibility trace factor
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
        epsilon is the initial threshold for epsilon greedy
        min_epsilon is the minimum value that epsilon should decay to
        epsilon_decay is the decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    '''
    init_epsilon = epsilon
    Et = np.zeros((Q.shape))
    for i in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon=epsilon)
        for j in range(max_steps):
            Et = Et * lambtha * gamma
            Et[state, action] += 1.0
            new_state, reward, done, info = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon=epsilon)
            # check whether get in hole or not and update rewards
            if env.desc.reshape(env.observation_space.n)[new_state] == b'H':
                reward = -1
            if env.desc.reshape(env.observation_space.n)[new_state] == b'G':
                reward = 1
            # deltat = R(t+ 1) + gamma * q(St+1,At+1) - q(St,At)
            deltat = reward + gamma * Q[new_state, new_action] -\
                Q[state, action]
            # Q(St) = Q(St) + alpha * deltat * Et(St)
            Q[state, action] = Q[state, action] + alpha * deltat *\
                Et[state, action]
            if done:
                break
            state = new_state
            action = new_action
        epsilon = (min_epsilon + (init_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * i))
    return Q
