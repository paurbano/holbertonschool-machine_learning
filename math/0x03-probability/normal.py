#!/usr/bin/env python3
'''normal distribution '''


class Normal():
    '''class normal distribution'''
    def __init__(self, data=None, mean=0., stddev=1.):
        '''constructor'''
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            add = sum([((x - self.mean) ** 2) for x in data])
            self.stddev = float((add / len(data)) ** 0.5)

    def z_score(self, x):
        '''calculate z-zcore for x '''
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        ''' calculate x-value for z-score'''
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        '''calculate pdf for x'''
        pi = 3.1415926536
        e = 2.7182818285
        factor = float(1 / ((self.stddev) * (2 * pi) ** 0.5))
        print(factor)
        exp = -0.5 * ((x - self.mean / self.stddev) ** 2)
        print(exp)
        return float(factor * (e ** exp))
