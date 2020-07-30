#!/usr/bin/env python3
'''exponential distribution '''


class Exponential():
    '''class exponential '''
    def __init__(self, data=None, lambtha=1.):
        '''constructor '''
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / float(sum(data) / len(data))

    def pdf(self, x):
        '''calculate pdf function'''
        if x < 0:
            return 0
        e = 2.7182818285
        pdf = self.lambtha * pow(e, -(self.lambtha) * x)
        return pdf

    def cdf(self, x):
        '''calculate cdf function '''
        if x < 0:
            return 0
        e = 2.7182818285
        cdf = 1 - pow(e, -(self.lambtha) * x)
        return cdf
