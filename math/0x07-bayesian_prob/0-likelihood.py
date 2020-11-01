#!/usr/bin/env python3
'''Maximum-likelihood (ML) Estimation
https://www.open.edu/openlearn/ocw/pluginfile.php/1066922/mod_resource/content/
       5/Modelling%20and%20estimation%20PDF.pdf
https://online.stat.psu.edu/stat504/node/28/

L(p;x) = (n!/x!(n−x)!) * ((p ** x) * (1−p)**(n−x))
'''

import numpy as np


def likelihood(x, n, P):
    '''calculates the likelihood
    Args:
        x: is the number of patients that develop severe side effects
        n: is the total number of patients observed
        P: is a 1D numpy.ndarray containing the various hypothetical
            probabilities of developing severe side effects
    Returns: a 1D numpy.ndarray containing the likelihood of obtaining
            the data, x and n, for each probability in P, respectively
    '''
    if type(n) is not int and n > 0:
        raise ValueError('n must be a positive integer')
    if type(x) is not int and x >= 0:
        raise ValueError('x must be an integer that is greater than or\
                         equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) is not np.ndarray:
        raise TypeError('P must be a 1D numpy.ndarray')
    if any(P < 0) or any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')
    L = np.math.factorial(n) / (np.math.factorial(x) *
                                np.math.factorial(n - x)) *\
        pow(P, x) * pow((1 - P), (n - x))
    return L
