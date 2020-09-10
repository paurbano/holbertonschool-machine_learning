#!/usr/bin/env python3
'''back propagation over a pooling layer'''

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''performs back propagation over a pooling layer of a neural network
    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
          partial derivatives with respect to the output of the pooling layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c is the number of channels
        A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing
                the output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
        kernel_shape: is a tuple of (kh, kw) containing the size of the kernel
                    for the pooling
            kh is the kernel height
            kw is the kernel width
        stride is a tuple of (sh, sw) containing the strides for the pooling
            sh is the stride for the height
            sw is the stride for the width
        mode: is a string containing either max or avg, indicating whether to
              perform maximum or average pooling, respectively
        Returns: the partial derivatives with respect to the previous layer
                (dA_prev)
    '''
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    sh, sw = stride
    kh, kw = kernel_shape

    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):                       # loop over the training examples
        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i]
        for h in range(n_H):               # loop on the vertical axis
            for w in range(n_W):           # loop on the horizontal axis
                for c in range(n_C):       # loop over the channels (depth)
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        # Use the corners and "c" to define the current slice
                        a_prev_slice = a_prev[vert_start:vert_end,
                                              horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        # Set dA_prev to be dA_prev
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] +=\
                            np.multiply(mask, dA[i, h, w, c])

                    elif mode == "avg":
                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = kernel_shape
                        average = da / (kw * kw)
                        a = np.ones(shape) * average
                        # Distribute it to get the correct slice of dA_prev.
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += a

    return dA_prev
