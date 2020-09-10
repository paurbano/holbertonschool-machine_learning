#!/usr/bin/env python3
'''propagation over a pooling layer'''

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''performs pooling on images
    Args:
        A_prev:array shape (m,h_prev,w_prev,c_prev) containing multiple
              grayscale images
            m is the number of images
            h_prev is the height in pixels of the images
            w_prev is the width in pixels of the images
            c_prev is the number of channels in the image
        kernel_shape: is a tuple of (kh, kw) containing the kernel
            kh is the height of the kernel
            kw is the width of the kernel
        stride is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        mode indicates the type of pooling
            max indicates max pooling
            avg indicates average pooling
        Return :  the output of the pooling layer
    '''
    m = A_prev.shape[0]
    h = A_prev.shape[1]
    w = A_prev.shape[2]
    c = A_prev.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    # define type of pooling
    if mode == 'max':
        pooling = np.max
    else:
        pooling = np.average
    # with pooling no padding is used
    # new dimensions with stride
    nh = int(((h - kh) / sh)) + 1
    nw = int(((w - kw) / sw)) + 1
    # output must contain channels
    conv = np.zeros((m, nh, nw, c))
    # Loop over every pixel of the output
    for i in range(nh):
        for j in range(nw):
            # apply strided
            x = i * sh
            y = j * sw
            # apply pooling to slice
            conv[:, i, j, :] = pooling(A_prev[:, x:x+kh, y:y+kw, :],
                                       axis=(1, 2))
    return conv
