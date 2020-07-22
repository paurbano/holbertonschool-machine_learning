#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
# print(fruit)

# your code here
labels = ['Farrah', 'Fred', 'Felicia']
width = 0.5
apples = fruit[0].copy()
bananas = fruit[1].copy()
oranges = fruit[2].copy()
peaches = fruit[3].copy()

fig, ax = plt.subplots()
ax.bar(labels, apples, width, label='apples', color='red')
ax.bar(labels, bananas, width, label='bananas', bottom=apples, color='yellow')
ax.bar(labels, oranges, width, label='oranges', bottom=bananas, color='#ff8000')
ax.bar(labels, peaches, width, label='peaches', bottom=oranges, color='#ffe5b4')


# ax.set_xticks(labels)
ax.set_yticks(range(0,90,10))
ax.set(ylim=(0,80))
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.legend()
plt.show()
