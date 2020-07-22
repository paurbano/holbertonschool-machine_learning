#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
fig = plt.figure()
fig.suptitle("All in One", fontsize=14)
grid = gridspec.GridSpec(nrows=3, ncols=2, hspace=1.0, wspace=0.4)

# figure task 0
ax0 = fig.add_subplot(grid[0, 0])
ax0.plot(y0, color='red')
ax0.set(xlim=(0, len(y0)-1))

# figure task 1
ax1 = fig.add_subplot(grid[0, 1])
ax1.plot(x1, y1, 'om')
ax1.set_title('Men\'s Height vs Weight', fontsize='x-small')
ax1.set_xlabel('Height (in)', fontsize=8)
ax1.set_ylabel('Weight (lbs)', fontsize=8)

# figure task 2
ax2 = fig.add_subplot(grid[1, 0])
ax2.set_xlabel('Time (years)', fontsize=8)
ax2.set_ylabel('Fraction Remaining', fontsize=8)
# title
ax2.set_title('Exponential Decay of C-14', fontsize=8)
# scale for y-axis
ax2.set(yscale=('log'))
# range for x-axis
ax2.set(xlim=(0, 28650))
ax2.plot(x2, y2)

# figure task 3
ax3 = fig.add_subplot(grid[1, 1])
ax3.plot(x3, y31, '--r', label='C-14')
ax3.plot(x3, y32, 'g', label='Ra-226')
ax3.legend(fontsize='xx-small')
ax3.set_xlabel('Time (years)', fontsize=8)
ax3.set_ylabel('Fraction Remaining', fontsize=8)
ax3.set_title('Exponential Decay of Radioactive Elements', fontsize='xx-small')
ax3.set(xlim=(0, 20000))
ax3.set(ylim=(0, 1))

#figure task 4
ax4 = fig.add_subplot(grid[2:, :])
num_bins = list(range(0,101,10))
# histogram of data
ax4.hist(student_grades, num_bins, edgecolor='black')
ax4.set(xlim=(0,100))
ax4.set(ylim=(0,30))
ax4.set_xticks(num_bins)
ax4.set_yticks(np.arange(0, 31, 10))
ax4.set_ylabel('Number of Students', fontsize='xx-small')
ax4.set_xlabel('Grades', fontsize='xx-small')
ax4.set_title('Project A', fontsize='x-small')


plt.show()
