# 0x03. Optimization

# General
* **What is a hyperparameter?**
    
* How and why do you normalize your input data?
* What is a saddle point?
* What is stochastic gradient descent?
* What is mini-batch gradient descent?
* What is a moving average? How do you implement it?
* What is gradient descent with momentum? How do you implement it?
* What is RMSProp? How do you implement it?
* What is Adam optimization? How do you implement it?
* What is learning rate decay? How do you implement it?
* What is batch normalization? How do you implement it?

# Tasks

## 0. Normalization Constants
Write the function `def normalization_constants(X):` that calculates the normalization (standardization) constants of a matrix:

    * X is the numpy.ndarray of shape (m, nx) to normalize
        * m is the number of data points
        * nx is the number of features
    * Returns: the mean and standard deviation of each feature, respectively

```
ubuntu@alexa-ml:~/0x03-optimization$ cat 0-main.py 
#!/usr/bin/env python3

import numpy as np
normalization_constants = __import__('0-norm_constants').normalization_constants

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    X = np.concatenate((a, b, c), axis=1)
    m, s = normalization_constants(X)
    print(m)
    print(s)
ubuntu@alexa-ml:~/0x03-optimization$ ./0-main.py 
[ 0.11961603  2.08201297 -3.59232261]
[2.01576449 1.034667   9.52002619]
ubuntu@alexa-ml:~/0x03-optimization
```
File : `0-norm_constants.py`

## 1. Normalize
Write the function `def normalize(X, m, s):` that normalizes (standardizes) a matrix:

* X is the numpy.ndarray of shape (d, nx) to normalize
    * `d` is the number of data points
    * `nx` is the number of features
* `m` is a `numpy.ndarray` of shape `(nx,)` that contains the mean of all features of `X`
* `s` is a `numpy.ndarray` of shape `(nx,)` that contains the standard deviation of all features of `X`
* Returns: The normalized `X` matrix

```
ubuntu@alexa-ml:~/0x03-optimization$ cat 1-main.py 
#!/usr/bin/env python3

import numpy as np
normalization_constants = __import__('0-norm_constants').normalization_constants
normalize = __import__('1-normalize').normalize

if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))
    X = np.concatenate((a, b, c), axis=1)
    m, s = normalization_constants(X)
    print(X[:10])
    X = normalize(X, m, s)
    print(X[:10])
    m, s = normalization_constants(X)
    print(m)
    print(s)
ubuntu@alexa-ml:~/0x03-optimization$ ./1-main.py 
[[  3.52810469   3.8831507   -6.69181838]
 [  0.80031442   0.65224094  -5.39379178]
 [  1.95747597   0.729515     7.99659596]
 [  4.4817864    2.96939671   3.55263731]
 [  3.73511598   0.82687659   3.40131526]
 [ -1.95455576   3.94362119 -19.16956044]
 [  1.90017684   1.58638102  -3.24326124]
 [ -0.30271442   1.25254519 -10.38030909]
 [ -0.2064377    3.92294203  -0.20075401]
 [  0.821197     3.48051479  -3.9815039 ]]
[[ 1.69091612  1.74078977 -0.32557639]
 [ 0.33768746 -1.38186686 -0.18922943]
 [ 0.91174338 -1.3071819   1.21732003]
 [ 2.16402779  0.85765153  0.75051893]
 [ 1.79361228 -1.21308245  0.73462381]
 [-1.02897526  1.79923417 -1.63625998]
 [ 0.88331787 -0.47902557  0.03666601]
 [-0.20951378 -0.80167608 -0.71302183]
 [-0.1617519   1.77924787  0.35625623]
 [ 0.34804709  1.35164437 -0.04088028]]
[ 2.44249065e-17 -4.99600361e-16  1.46549439e-16]
[1. 1. 1.]
ubuntu@alexa-ml:~/0x03-optimization$
```
File: `1-normalize.py`
