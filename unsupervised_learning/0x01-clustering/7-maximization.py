#!/usr/bin/env python3
'''Maximization '''
import numpy as np


def maximization(X, g):
    '''calculates the maximization step in the EM algorithm for a GMM
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the data set
        g: is a numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster
    Returns: pi, m, S, or None, None, None on failure
        pi: is a numpy.ndarray of shape (k,) containing the updated priors for
            each cluster
        m: is a numpy.ndarray of shape (k, d) containing the updated centroid
            means for each cluster
        S: is a numpy.ndarray of shape (k, d, d) containing the updated
            covariance matrices for each cluster
    '''
    if type(X) is not np.ndarray or type(g) is not np.ndarray:
        return (None, None, None)

    if len(X.shape) != 2 or len(g.shape) != 2:
        return (None, None, None)

    if X.shape[0] != g.shape[1]:
        return (None, None, None)

    k = g.shape[0]
    n, d = X.shape
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return (None, None, None)
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        m[i] = np.sum(g[i, :, None] * X, axis=0) / np.sum(g[i], axis=0)
        aux = X - m[i]
        S[i] = np.dot(g[i] * aux.T, aux) / np.sum(g[i])
        pi[i] = np.sum(g[i]) / n
    return pi, m, S
