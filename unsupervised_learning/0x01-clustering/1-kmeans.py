#!/usr/bin/env python3
'''K-means
https://medium.com/analytics-vidhya/k-means-clustering-with-python-77b20c2d538d
https://www.youtube.com/watch?v=_aWzGGNrcic
'''

import numpy as np


def kmeans(X, k, iterations=1000):
    ''' performs K-means on a dataset
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset
            n is the number of data points
            d is the number of dimensions for each data point
        k: is a positive integer containing the number of clusters
        iterations: is a positive integer containing the maximum number of
                    iterations that should be performed
    Returns: C, clss, or None, None on failure
        C: is a numpy.ndarray of shape (k, d) containing the centroid means
           for each cluster
        clss: is a numpy.ndarray of shape (n,) containing the index of the
              cluster in C that each data point belongs to
    '''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    # initialize centroids with multivariate uniform distribution
    n, d = X.shape
    _min = np.amin(X, axis=0)
    _max = np.amax(X, axis=0)
    C = np.random.uniform(_min, _max, size=(k, d))
    #
    for i in range(iterations):
        # find all the points to the nearest cluster centroid
        # calculating Euclidean distance
        clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
        newCentroid = np.copy(C)
        # Recompute position for every centroids for every cluster
        for c in range(k):
            # If a cluster contains no data points during the update step
            # reinitialize its centroid
            # else recompute centroid with average of all points
            if c not in clss:
                newCentroid[c] = np.random.uniform(_min, _max)
            else:
                newCentroid[c] = np.mean(X[clss == c], axis=0)
        # if Centroids of newly formed clusters do not change return
        # else assign new centroids to recompute it
        if np.array_equal(newCentroid, C):
            return (C, clss)
        else:
            C = newCentroid
    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
    return (C, clss)
