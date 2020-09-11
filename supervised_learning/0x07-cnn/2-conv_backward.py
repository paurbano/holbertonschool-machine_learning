#!/usr/bin/env python3
'''back propagation over a Convolutional layer'''

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    '''performs back propagation over a convolutional layer of a neural network
       Args:
            dZ:numpy.ndarray of shape (m, h_new, w_new, c_new)
                  containing the output of the previous layer
                m is the number of examples
                h_new is the height of the output
                w_new is the width of the output
                c_new is the number of channels in the output
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
            padding: is a string that is either same or valid, indicating the
                    type of padding used
            stride:tuple of (sh, sw) containing the strides for the convolution
                sh is the stride for the height
                sw is the stride for the width
            Returns: the partial derivatives with respect to the previous layer
                (dA_prev), the kernels (dW), and the biases (db), respectively
    '''
    # Retrieve dimensions from A_prev's shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (kh, kw, c_prev, c_new) = W.shape

    # stride
    sh, sw = stride

    if padding == 'same':
        padh = int((((h_prev - 1) * sh - h_prev + kh) / 2)) + 1
        padw = int((((w_prev - 1) * sw - w_prev + kw) / 2)) + 1
    else:
        padh = padw = 0

    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    # db = np.zeros((1, 1, 1, n_C))
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    # Pad A_prev and dA_prev
    pad = ((0, 0), (padh, padh), (padw, padw), (0, 0))
    A_prev_pad = np.pad(A_prev, pad_width=pad, mode='constant',
                        constant_values=0)
    dA_prev_pad = np.pad(dA_prev, pad_width=pad, mode='constant',
                         constant_values=0)

    # loop over the training examples
    for i in range(m):
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        # loop over vertical(height) axis of the output volume
        for h in range(n_H):
            # loop over horizontal axis of the output volume
            for w in range(n_W):
                # loop over the channels of the output volume
                for c in range(n_C):
                    # Find the corners of the current "slice"
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end]

                    # Update gradients for the window and the filter's
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end]\
                        += W[:, :, :, c] * dZ[i, h, w, c]

                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    # db[:, :, :, c] += dZ[i, h, w, c]

        # why ?
        if padding == 'same':
            dA_prev[i, :, :, :] += da_prev_pad[padh:-padh, padw:-padw, :]
        if padding == 'valid':
            dA_prev[i, :, :, :] += da_prev_pad
    return dA_prev, dW, db
