# 0x01. Clustering

## Learning Objectives
* What is a multimodal distribution?
* What is a cluster?
* 9hat is cluster analysis?
* What is “soft” vs “hard” clustering?
* What is K-means clustering?
* What are mixture models?
* What is a Gaussian Mixture Model (GMM)?
* What is the Expectation-Maximization (EM) algorithm?
* How to implement the EM algorithm for GMMs
* What is cluster variance?
* What is the mountain/elbow method?
* What is the Bayesian Information Criterion?
* How to determine the correct number of clusters
* What is Hierarchical clustering?
* What is Agglomerative clustering?
* What is Ward’s method?
* What is Cophenetic distance?
* What is scikit-learn?
* What is scipy?

## Installing Scikit-Learn 0.21.x
```
pip install --user scikit-learn==0.21
```
## Installing Scipy 1.3.x
scipy should have already been installed with matplotlib and numpy, but just in case:
```
pip install --user scipy==1.3
```

## 0. Initialize K-means
Write a function `def initialize(X, k):` that initializes cluster centroids for K-means:

* `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset that will be used for K-means clustering
    * `n` is the number of data points
    * `d` is the number of dimensions for each data point
* `k` is a positive integer containing the number of clusters
* The cluster centroids should be initialized with a multivariate uniform distribution along each dimension in d:
    * The minimum values for the distribution should be the minimum values of X along each dimension in d
    * The maximum values for the distribution should be the maximum values of X along each dimension in d
    * You should use `numpy.random.uniform` exactly once
* You are not allowed to use any loops
* Returns: a `numpy.ndarray` of `shape (k, d)` containing the initialized centroids for each cluster, or `None` on failure
```
alexa@ubuntu-xenial:0x01-clustering$ cat 0-main.py 
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
initialize = __import__('0-initialize').initialize

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.show()
    print(initialize(X, 5))
alexa@ubuntu-xenial:0x01-clustering$ ./0-main.py 
```
```
[[14.54730144 13.46780434]
 [20.57098466 33.55245039]
 [ 9.55556506 51.51143281]
 [48.72458008 20.03154959]
 [25.43826106 60.35542243]]
alexa@ubuntu-xenial:0x01-clustering$
```

## 1. K-means
Write a function `def kmeans(X, k, iterations=1000):` that performs K-means on a dataset:

* X is a `numpy.ndarray` of shape `(n, d)` containing the dataset
    * `n` is the number of data points
    * `d` is the number of dimensions for each data point
* k is a positive integer containing the number of clusters
* iterations is a positive integer containing the maximum number of iterations that should be performed
* If no change in the cluster centroids occurs between iterations, your function should return
* Initialize the cluster centroids using a multivariate uniform distribution (based on0-initialize.py)
* If a cluster contains no data points during the update step, reinitialize its centroid
* You should use numpy.random.uniform exactly twice
* You may use at most 2 loops
* Returns: `C`, clss, or None, None on failure
    * C is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster
    * clss is a `numpy.ndarray` of shape `(n,)` containing the index of the cluster in C that each data point belongs to
```
alexa@ubuntu-xenial:0x01-clustering$ cat 1-main.py 
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
kmeans = __import__('1-kmeans').kmeans

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    C, clss = kmeans(X, 5)
    print(C)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
    plt.show()
alexa@ubuntu-xenial:0x01-clustering$ ./0-main.py 
[[ 9.92511389 25.73098987]
 [30.06722465 40.41123947]
 [39.62770705 19.89843487]
 [59.22766628 29.19796006]
 [20.0835633  69.81592298]]
```
