#!/usr/bin/env python3
''' calculates the derivative of a polynomial '''


def poly_derivative(poly):
    '''calculates the derivative of a polynomial'''

    if not isinstance(poly, list) or poly is None or poly == []:
        return None

    if all(isinstance(i, (int, float)) for i in poly):
        True
    else:
        return None

    derivative = [poly[i] * i for i in range(len(poly))]

    if len(derivative) == 1:
        return derivative
    else:
        return derivative[1:]
