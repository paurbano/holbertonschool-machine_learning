#!/usr/bin/env python3
''' sum squares'''


def summation_i_squared(n):
    '''sum squares '''
    sum = 0
    if type(n) is not int or n < 1:
        return None
    if n == 1:
        return 1

    return (summation_i_squared(n-1) + n**2)
