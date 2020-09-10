#!/usr/bin/env python3
'''Convolutional Forward Propagation'''

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    '''performs forward propagation over a convolutional layer of a neural
       network:
       Args:
            A_prev:numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                  containing the output of the previous layer
                m is the number of examples
                h_prev is the height of the previous layer
                w_prev is the width of the previous layer
                c_prev is the number of channels in the previous layer
            W: is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
               the kernels for the convolution
                kh is the filter height
                kw is the filter width
                c_prev is the number of channels in the previous layer
                c_new is the number of channels in the output
            b: is a numpy.ndarray of shape (1, 1, 1, c_new) containing the
                biases applied to the convolution
            activation: is an activation function applied to the convolution
            padding: is a string that is either same or valid, indicating the
                    type of padding used
            stride:tuple of (sh, sw) containing the strides for the convolution
                sh is the stride for the height
                sw is the stride for the width
            Returns: the output of the convolutional layer
    '''
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        padh = int((((h_prev - 1) * sh - h_prev + kh) / 2))
        padw = int((((w_prev - 1) * sw - w_prev + kw) / 2))
    else:
        padh = padw = 0

    # dimensions for CONV output
    nh = int(((h_prev + (2 * padh) - kh) / sh)) + 1
    nw = int(((w_prev + (2 * padw) - kw) / sw)) + 1
    # Initialize output volume Z and activations A
    Z = np.zeros([m, nh, nw, c_new])
    # Add Padding to input: A_prev
    pad = ((0, 0), (padh, padh), (padw, padw), (0, 0))
    A_prev_pad = np.pad(A_prev, pad_width=pad, mode='constant',
                        constant_values=0)

    # loop over vertical axis of the output volume
    for h in range(nh):
        # loop over horizontal axis of the output
        for w in range(nw):
            # loop over channels(= #filters of layer) of the output
            for c in range(c_new):
                # get position filter according to stride
                x = h * sh
                y = w * sw
                # slice a_prev_pad images with filter
                a_slice_pad = A_prev_pad[:, x:x+kh, y:y+kw, :]
                # convolve the slice with filter W get Z output
                Z[:, h, w, c] = np.multiply(a_slice_pad, W[:, :, :, c]).\
                    sum(axis=(1, 2, 3))
    Z = Z + b
    # Apply activation
    return activation(Z)
