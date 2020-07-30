#!/usr/bin/env python3
''' class poisson '''


class Poisson():
    ''' Poisson'''
    def __init__(self, data=None, lambtha=1.):
        ''' Constructor'''
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        '''Calculates PMF '''
        if k < 0:
            # print('evaluo k')
            return 0
        k = int(k)
        e = 2.7182818285
        factorial = 1
        for i in range(1, int(k) + 1):
            factorial = factorial * i

        return (pow(e, -self.lambtha) * (pow(self.lambtha, k)) / factorial)

    def cdf(self, k):
        '''calculate CDF '''
        if k < 0:
            # print('evaluo k')
            return 0
        cumulative = 0
        for x in range(0, int(k) + 1):
            cumulative += self.pmf(x)

        return cumulative
