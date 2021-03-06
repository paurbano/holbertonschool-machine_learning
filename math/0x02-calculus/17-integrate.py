#!/usr/bin/env python3
'''integrate '''


def poly_integral(poly, C=0):
    '''calculates the integral of a polynomial:'''
    if not isinstance(poly, list) or poly is None or poly == []:
        return None

    if all(isinstance(i, (int, float)) for i in poly):
        True
    else:
        return None

    if not isinstance(C, (int, float)) or C is None:
        return None

    if all(isinstance(i, (int, float)) for i in poly):
        True
    else:
        return None
    # when is just the constant
    if poly == [0]:
        return [C]

    poly.insert(0, C)
    integral = [poly[i] / i for i in range(len(poly)) if i > 0]
    # del poly[0]
    for coeff in range(len(integral)):
        fraction = integral[coeff] % 1
        if fraction == 0:
            integral[coeff] = round(integral[coeff])
    integral.insert(0, C)
    return integral
