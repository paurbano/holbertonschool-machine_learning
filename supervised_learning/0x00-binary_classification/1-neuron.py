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
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''getter for Weigth'''
        return self.__W

    @property
    def b(self):
        '''getter for bias'''
        return self.__b

    @property
    def A(self):
        '''getter for activited function'''
        return self.__A
