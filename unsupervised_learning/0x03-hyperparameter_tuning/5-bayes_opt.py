#!/usr/bin/env python3
'''Initialize Bayesian Optimization
https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/
dev/bayesian-optimization/bayesian_optimization.ipynb
'''

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    '''performs Bayesian optimization on a noiseless 1D Gaussian process'''
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        '''Constructor
        Args:
            f: is the black-box function to be optimized
            X_init: is a numpy.ndarray of shape (t, 1) representing
                the inputs already sampled with the black-box function
            Y_init: is a numpy.ndarray of shape (t, 1) representing
            the outputs of the black-box function for each input in X_init
            t: is the number of initial samples
            bounds: is a tuple of (min, max) representing the bounds
                of the space in which to look for the optimal point
            ac_samples: is the number of samples that should be
                        analyzed during acquisition
            l: is the length parameter for the kernel
            sigma_f: is the standard deviation given to the output of
                    the black-box function
            xsi: is the exploration-exploitation factor for acquisition
            minimize: is a bool determining whether optimization should
            be performed for minimization (True) or maximization (False)
        '''
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        '''calculates the next best sample location
        Uses the Expected Improvement acquisition function
        Returns: X_next, EI
            X_next: is a numpy.ndarray of shape (1,) representing the next best
                    sample point
            EI: is a numpy.ndarray of shape (ac_samples,) containing the
                expected improvement of each potential sample
        '''
        #  GP: xt=argmax_x u(x|D1:t−1)
        mu, sigma = self.gp.predict(self.X_s)
        with np.errstate(divide='warn'):
            if self.minimize:
                mu_sample_opt = np.min(self.gp.Y)
                improvement = (mu_sample_opt - mu - self.xsi)
            else:
                mu_sample_opt = np.amax(self.gp.Y)
                improvement = (mu - mu_sample_opt - self.xsi)
            # (2)
            Z = improvement / sigma
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        '''optimizes the black-box function
        Args:
            iterations is the maximum number of iterations to perform
        Returns: X_opt, Y_opt
            X_opt: is a numpy.ndarray of shape (1,) representing the optimal
                    point
            Y_opt: is a numpy.ndarray of shape (1,) representing the optimal
                    function value
        '''
        for i in range(iterations):
            X_next, _ = self.acquisition()
            Y_next = self.f(X_next)
            if (X_next == self.gp.X).any():
                self.gp.X = self.gp.X[:-1]
                break
            self.gp.update(X_next, Y_next)
        if self.minimize:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)
        Y_opt = self.gp.Y[index]
        X_opt = self.gp.X[index]
        return (X_opt, Y_opt)
