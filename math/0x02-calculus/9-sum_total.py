#!/usr/bin/env python3
''' sum squares'''
from functools import reduce


def summation_i_squared(n):
    '''sum squares '''

    if (type(n) is not int or n < 1):
        return None
    # return (summation_i_squared(n-1) + n**2)
    return round((n * (n + 1) * ((2 * n) + 1)) / 6)
