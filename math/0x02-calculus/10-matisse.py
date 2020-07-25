#!/usr/bin/env python3
''' calculates the derivative of a polynomial '''


def poly_derivative(poly):
    '''calculates the derivative of a polynomial'''
    derivative = [0]
    if not isinstance(poly, list):
        return None

    derivative = [poly[i] * i for i in range(len(poly))]

    if len(derivative) == 1:
        return derivative
    else:
        return derivative[1:]
