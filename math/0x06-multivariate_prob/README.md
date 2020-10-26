# 0x06. Multivariate Probability

## General
* **Who is Carl Friedrich Gauss?**
* **What is a joint/multivariate distribution?**
* **What is a covariance?**
* **What is a correlation coefficient?**
* **What is a covariance matrix?**
* **What is a multivariate Gaussian distribution?**

# Tasks

## 0. Mean and Covariance
Write a function `def mean_cov(X):` that calculates the mean and covariance of a data set:

* X is a `numpy.ndarray` of shape `(n, d)` containing the data set:
    * `n` is the number of data points
    * `d` is the number of dimensions in each data point
    * If `X` is not a 2D `numpy.ndarray`, raise a `TypeError` with the message `X must be a 2D numpy.ndarray`
    * If n is less than 2, raise a `ValueError` with the message `X must contain multiple data points`
* Returns: `mean, cov`:
    * `mean` is a `numpy.ndarray` of shape `(1, d)` containing the mean of the data set
    * `cov` is a `numpy.ndarray` of shape `(d, d)` containing the covariance matrix of the data set
* You are not allowed to use the function `numpy.cov`
```
alexa@ubuntu-xenial:0x06-multivariate_prob$ cat 0-main.py
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    mean_cov = __import__('0-mean_cov').mean_cov

    np.random.seed(0)
    X = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000)
    mean, cov = mean_cov(X)
    print(mean)
    print(cov)
alexa@ubuntu-xenial:0x06-multivariate_prob$ ./0-main.py 
[[12.04341828 29.92870885 10.00515808]]
[[ 36.2007391  -29.79405239  15.37992641]
 [-29.79405239  97.77730626 -20.67970134]
 [ 15.37992641 -20.67970134  24.93956823]]
alexa@ubuntu-xenial:0x06-multivariate_prob$
```
File: [0-mean_cov.py]

## 1. Correlation
Write a function `def correlation(C):` that calculates a correlation matrix:

* C is a numpy.ndarray of shape (d, d) containing a covariance matrix
    * d is the number of dimensions
    * If C is not a numpy.ndarray, raise a TypeError with the message C must be a numpy.ndarray
    * If C does not have shape (d, d), raise a ValueError with the message C must be a 2D square matrix
* Returns a numpy.ndarray of shape (d, d) containing the correlation matrix
```
alexa@ubuntu-xenial:0x06-multivariate_prob$ cat 1-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    correlation = __import__('1-correlation').correlation

    C = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    Co = correlation(C)
    print(C)
    print(Co)
alexa@ubuntu-xenial:0x06-multivariate_prob$ ./1-main.py 
[[ 36 -30  15]
 [-30 100 -20]
 [ 15 -20  25]]
[[ 1.  -0.5  0.5]
 [-0.5  1.  -0.4]
 [ 0.5 -0.4  1. ]]
alexa@ubuntu-xenial:0x06-multivariate_prob$
```
File: [1-correlation.py]

## 2. Initialize
Create the class MultiNormal that represents a Multivariate Normal distribution:

`class constructor def __init__(self, data):`
data is a numpy.ndarray of shape (d, n) containing the data set:
n is the number of data points
d is the number of dimensions in each data point
If data is not a 2D numpy.ndarray, raise a TypeError with the message data must be a 2D numpy.ndarray
If n is less than 2, raise a ValueError with the message data must contain multiple data points
Set the public instance variables:
mean - a numpy.ndarray of shape (d, 1) containing the mean of data
cov - a numpy.ndarray of shape (d, d) containing the covariance matrix data
You are not allowed to use the function numpy.cov
```
alexa@ubuntu-xenial:0x06-multivariate_prob$ cat 2-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    from multinormal import MultiNormal

    np.random.seed(0)
    data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    print(mn.mean)
    print(mn.cov)
alexa@ubuntu-xenial:0x06-multivariate_prob$ ./2-main.py 
[[12.04341828]
 [29.92870885]
 [10.00515808]]
[[ 36.2007391  -29.79405239  15.37992641]
 [-29.79405239  97.77730626 -20.67970134]
 [ 15.37992641 -20.67970134  24.93956823]]
alexa@ubuntu-xenial:0x06-multivariate_prob$
```
File: [multinormal.py]

## 3. PDF
Update the class MultiNormal:

public instance method def pdf(self, x): that calculates the PDF at a data point:
x is a numpy.ndarray of shape (d, 1) containing the data point whose PDF should be calculated
d is the number of dimensions of the Multinomial instance
If x is not a numpy.ndarray, raise a TypeError with the message x must be a numpy.ndarray
If x is not of shape (d, 1), raise a ValueError with the message x must have the shape ({d}, 1)
Returns the value of the PDF
You are not allowed to use the function numpy.cov
```
alexa@ubuntu-xenial:0x06-multivariate_prob$ cat 3-main.py 
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    from multinormal import MultiNormal

    np.random.seed(0)
    data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    x = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 1).T
    print(x)
    print(mn.pdf(x))
alexa@ubuntu-xenial:0x06-multivariate_prob$ ./3-main.py 
[[ 8.20311936]
 [32.84231319]
 [ 9.67254478]]
0.00022930236202143824
alexa@ubuntu-xenial:0x06-multivariate_prob$
```
File: [multinormal.py]