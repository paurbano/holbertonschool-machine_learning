#!/usr/bin/env python3
''' My first Neuron'''

import numpy as np


class Neuron():
    '''class Neuron'''
    def __init__(self, nx):
        '''constructor'''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
