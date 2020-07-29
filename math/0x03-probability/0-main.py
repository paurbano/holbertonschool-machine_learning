#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
data1 = []
p1 = Poisson(data)
print('Lambtha:', p1.lambtha)

p2 = Poisson(lambtha=5)
print('Lambtha:', p2.lambtha)
'''
p3 = Poisson(data1)
print('Lambtha:', p3.lambtha)

d=2
p4 = Poisson(d)
print('Lambtha:', p4.lambtha)
'''
p5 = Poisson(lambtha=-5)
print('Lambtha:', p5.lambtha)