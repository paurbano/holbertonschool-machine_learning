#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3
# plot with line red
plt.plot(y, color='red')
# adjust x-axis scale start 0,0
plt.xlim(0, len(y)-1)
plt.show()
