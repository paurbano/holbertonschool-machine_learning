#!/usr/bin/env python3
'''deep Q-learning'''

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    '''class atari'''
    def process_observation(self, observation):
        '''process observation'''
        # (height, width, channel)
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        # resize and convert to grayscale
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        '''
        We could perform this processing step in `process_observation`.
        In this case, however, we would need to store a `float32` array
        instead, which is 4x more memory intensive than an `uint8` array.
        This matters if we store 1M observations.
        '''
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        '''clip reward'''
        return np.clip(reward, -1., 1.)


# Get the environment and extract the number of actions.
env = gym.make('Breakout-v0')
np.random.seed(123)
env.seed(123)
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

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=.05, nb_steps=10000)
processor = AtariProcessor()

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
               memory=memory, processor=processor, nb_steps_warmup=50000,
               gamma=.99, target_model_update=10000, train_interval=4)

dqn.compile(Adam(lr=.00025), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False)
# After training is done, we save the final weights one more time.
weights_filename = 'policy.h5'
dqn.save_weights(weights_filename, overwrite=True)
