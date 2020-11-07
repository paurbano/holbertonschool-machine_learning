#!/usr/bin/env python3
'''optimum number of clusters by variance'''

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    '''tests for the optimum number of clusters by variance
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the data set
        kmin: is a positive integer containing the minimum number of clusters
              to check for (inclusive)
        kmax: is a positive integer containing the maximum number of clusters
              to check for (inclusive)
        iterations: is a positive integer containing the maximum number of
                    iterations for K-means
    Returns: results, d_vars, or None, None on failure
        results: is a list containing the outputs of K-means for each cluster
                 size
        d_vars: is a list containing the difference in variance from the
                smallest cluster size for each cluster size
    '''
    if type(X) is not np.ndarray:
        return (None, None)

    if type(kmin) is not int:
        return (None, None)

    if kmax is not None and type(kmax) is not int:
        return (None, None)

    if kmax is None:
        kmax = X.shape[0]

    if len(X.shape) != 2 or kmin < 1:
        return (None, None)

    if kmax is not None and kmax <= kmin:
        return (None, None)

    if type(iterations) is not int:
        return (None, None)

    if iterations <= 0:
        return (None, None)

    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        var = variance(X, results[0][0]) - variance(X, C)
        d_vars.append(var)
    return results, d_vars
