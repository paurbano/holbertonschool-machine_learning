#!/usr/bin/env python3
'''deep Q-learning'''

import gym
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from keras.optimizers import Adam

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
weights_filename = 'policy.h5'
env = gym.make('Breakout-v0')
env.reset()
nb_actions = env.action_space.n

input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
model.add(Permute((2, 3, 1), input_shape=input_shape))
model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(nb_actions, activation='relu'))

'''
policy = LinearAnnealedPolicy(GreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05, nb_steps=10000)
'''
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=GreedyQPolicy(),
               memory=memory, processor=processor, nb_steps_warmup=500,
               gamma=.99, target_model_update=1e-2, train_interval=4,
               delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])
dqn.load_weights(weights_filename)
dqn.test(env, nb_episodes=10, visualize=True)
