#!/usr/bin/env python3
'''Initialize Gaussian Process
https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/
dev/gaussian-processes/gaussian_processes.ipynb?flush_cache=true
'''

import numpy as np


class GaussianProcess():
    '''represents a noiseless 1D Gaussian process'''
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        '''constructor
        Args:
            X_init: is a numpy.ndarray of shape (t, 1) representing the inputs
                   already sampled with the black-box function
            Y_init: is a numpy.ndarray of shape (t, 1) representing the outputs
                    of the black-box function for each input in X_init
            t: is the number of initial samples
            l: is the length parameter for the kernel
            sigma_f: is the standard deviation given to the output of the
                    black-box function
        '''
        if type(X_init) is not np.ndarray or len(X_init.shape) != 2:
            return None
        if type(Y_init) is not np.ndarray or len(Y_init.shape) != 2:
            return None
        if X_init.shape[0] != Y_init.shape[0]:
            return None
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        '''calculates the covariance kernel matrix between two matrices
        Args:
            X1 is a numpy.ndarray of shape (m, 1)
            X2 is a numpy.ndarray of shape (n, 1)
            kernel should use the Radial Basis Function (RBF)
        Returns: the covariance kernel matrix as a numpy.ndarray of shape(m, n)
        '''
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 *\
            np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        '''predicts the mean and standard deviation of points in a
            Gaussian process
        Args:
            X_s:is a numpy.ndarray of shape (s, 1) containing all of the points
                whose mean and standard deviation should be calculated
            s: is the number of sample points
        Returns: mu, sigma
            mu: is a numpy.ndarray of shape (s,) containing the mean for each
                point in X_s, respectively
            sigma: is a numpy.ndarray of shape (s,) containing the variance
                    for each point in X_s, respectively
        '''
        # Sigma
        sigma_y = 1e-8
        K = self.K + sigma_y**2 * np.eye(len(self.X))
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s) + 1e-8 * np.eye(len(X_s))
        K_inv = np.linalg.inv(K)
        # L = np.linalg.cholesky(K + 0.00005*np.eye(len(self.X)))
        # Lk = np.linalg.solve(L, K_s)

        mu = K_s.T.dot(K_inv).dot(self.Y).reshape((X_s.shape[0],))
        # s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
        s2 = K_ss - K_s.T.dot(K_inv).dot(K_s)
        return mu, np.diag(s2)

    def update(self, X_new, Y_new):
        '''updates a Gaussian Process:
        Args:
            X_new: is a numpy.ndarray of shape (1,) that represents the new
                    sample point
            Y_new: is a numpy.ndarray of shape (1,) that represents the new
                    sample function value
            Updates the public instance attributes X, Y, and K
        '''
        self.X = np.append(self.X, [X_new], axis=0)
        self.Y = np.append(self.Y, [Y_new], axis=0)
        self.K = self.kernel(self.X, self.X)
