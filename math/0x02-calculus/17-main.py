#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))

poly = [2, 6, '3', 2, 4]
print(poly_integral(poly))

poly = [3,0,3]
print(poly_integral(poly,0.25))

poly = [1,2,3,4,5]
print(poly_integral(poly,None))

poly = [1,2,3,4,5]
print(poly_integral(poly,5))