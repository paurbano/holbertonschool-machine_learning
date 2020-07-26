#!/usr/bin/env python3

poly_derivative = __import__('10-matisse').poly_derivative

poly = [5, 3, 0, 1]
print(poly_derivative(poly))

poly = [2, 6, 3, 2, 4]
print(poly_derivative(poly))

poly = [3,0,3]
print(poly_derivative(poly))

poly = [0, 6]
print(poly_derivative(poly))

poly = [6]
print(poly_derivative(poly))

poly = ['a']
print(poly_derivative(poly))