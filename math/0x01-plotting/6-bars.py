#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# y-axis in bold
# rc('font', weight='bold')

# Values of each group
apples = fruit[0].copy()
bananas = fruit[1].copy()
oranges = fruit[2].copy()
peaches = fruit[3].copy()

# Heights of bars
bars = np.add(apples, bananas).tolist()
bars2 = np.add(bars, oranges).tolist()
# The position of the bars on the x-axis
r = [0, 1, 2]

# Names of group and bar width
names = ['Farrah', 'Fred', 'Felicia']
width = 0.5

# Create apples bar
plt.bar(r, apples, width, color='red', label='apples')
# Create bananas bars (middle), on top of the firs ones
plt.bar(r, bananas, width, bottom=apples, color='yellow', label='bananas')
# Create oranges bars
plt.bar(r, oranges, width, bottom=bars, color='#ff8000', label='oranges')
# create peaches bars (top)
plt.bar(r, peaches, width, bottom=bars2, color='#ffe5b4', label='peaches')

# Custom X axis
plt.yticks(range(0, 90, 10))
plt.xticks(r, names)
plt.ylabel('Quantity of Fruit')
plt.suptitle('Number of Fruit per Person')
plt.legend()

# Show graphic
plt.show()
