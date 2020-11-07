#!/usr/bin/env python3
''' agglomerative clustering'''

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    '''performs agglomerative clustering on a dataset
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset
        dist: is the maximum cophenetic distance for all clusters
    Performs agglomerative clustering with Ward linkage
    Displays the dendrogram with each cluster displayed in a different color
    Returns: clss, a numpy.ndarray of shape (n,) containing the cluster
            indices for each data point
    '''
    Z = scipy.cluster.hierarchy.linkage(X, method="ward")
    dendrogram = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')
    plt.show()
    return clss
