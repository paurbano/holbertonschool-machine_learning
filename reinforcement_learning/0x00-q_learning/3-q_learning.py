#!/usr/bin/env python3
'''Q-learning
https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/
master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb
'''
import gym
import numpy as np
epsilonGreedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    '''performs Q-learning
    Args:
        env: is the FrozenLakeEnv instance
        Q: is a numpy.ndarray containing the Q-table
        episodes: is the total number of episodes to train over
        max_steps: is the maximum number of steps per episode
        alpha: is the learning rate
        gamma: is the discount rate
        epsilon: is the initial threshold for epsilon greedy
        min_epsilon: is the minimum value that epsilon should decay to
        epsilon_decay: is the decay rate for updating epsilon between episodes
    When the agent falls in a hole, the reward should be updated to be -1
    Returns: Q, total_rewards
        Q is the updated Q-table
        total_rewards is a list containing the rewards per episode
    '''
    # List of rewards
    rewards = []
    max_epsilon = epsilon
    # 2 For life or until learning is stopped
    for episode in range(episodes):
        # Reset the environment
        state = env.reset()
        # step = 0
        done = False
        t_rewards = 0

        for step in range(max_steps):
            action = epsilonGreedy(Q, state, epsilon)

            # Take the action (a) and observe the outcome state(s')
            # and reward (r)
            new_state, reward, done, info = env.step(action)
            # agent falls in a hole
            if done and reward == 0:
                reward = -1
            # Update Q(s,a):= Q(s,a)+lr [R(s,a) + gamma * max Q(s',a')-Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            Q[state, action] = Q[state, action] + alpha *\
                (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            t_rewards += reward

            # Our new state is state
            state = new_state

            # If done (if we're dead) : finish episode
            if done:
                break

        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) *\
            np.exp(-epsilon_decay * episode)
        rewards.append(t_rewards)
    return (Q, rewards)
