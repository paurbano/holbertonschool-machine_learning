#!/usr/bin/env python3
'''weighted moving average'''

import numpy as np


def moving_average(data, beta):
    '''calculates the weighted moving average of a data set
        data: list of data to calculate the moving average of
        beta: is the weight used for the moving average
        Returns: a list containing the moving averages of datas
    '''
    v = 0
    average = []
    for i in range(len(data)):
        # moving average
        v = (beta * v) + ((1 - beta) * data[i])
        # apply bias to get a better estimate
        mv = v / (1 - beta ** (i + 1))
        average.append(mv)
    return average
