#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
# your code here

num_bins = list(range(0,101,10))
# histogram of data
plt.hist(student_grades, num_bins, edgecolor='black')
plt.xlim(0,100)
plt.ylim(0,30)
plt.xticks(num_bins)
plt.ylabel('Number of Students')
plt.xlabel('Grades')
plt.suptitle('Project A')
plt.show()
