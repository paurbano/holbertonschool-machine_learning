#!/usr/bin/env python3
'''Simple Policy function'''
import numpy as np


def policy(matrix, weight):
    '''computes to policy with a weight of a matrix
    Args:
        Matrix: matrix of states
        weight: matrix of weights
    REturn:
    '''
    # for each column of weigth, sum wi*si
    z = matrix.dot(weight)
    # to accomplish request
    exp = np.exp(z)
    # return exp/sum(exp(z))
    return exp / np.sum(exp)


def policy_gradient(state, weight):
    '''computes the Monte-Carlo policy gradient based on a state and a
        weight matrix
    Args:
        state: matrix representing the current observation of the environment
        weight: matrix of random weight
    Return: the action and the gradient (in this order)
    '''
    # compute the probs
    P = policy(state, weight)
    # take an action randomly
    action = np.random.choice(len(P[0]), p=P[0])
    # compute the gradient, save it with reward to be able to update the
    # weigths
    # P looks like [P0, P1]; it's an array of array with one line
    s = P.reshape(-1, 1)
    softmax = np.diagflat(s) - np.dot(s, s.T)
    # take the obs dor action taken
    dsoftmax = softmax[action, :]
    # dlog
    dlog = dsoftmax / P[0, action]
    # upgrade the gradient
    grad = state.T.dot(dlog[None, :])
    return action, grad
