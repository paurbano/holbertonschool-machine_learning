#!/usr/bin/env python3
'''Pooling'''

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    '''performs pooling on images
    Args:
        images:array shape (m, h, w, c) containing multiple grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernel_shape: is a tuple of (kh, kw) containing the kernel
            kh is the height of the kernel
            kw is the width of the kernel
        padding is a tuple of (ph, pw) ‘same’, or ‘valid’
            if ‘same’, performs a same convolution
            if ‘valid’, performs a valid convolution
            if a tuple:
                ph is the padding for the height of the image
                pw is the padding for the width of the image
            the image should be padded with 0’s
        stride is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        mode indicates the type of pooling
            max indicates max pooling
            avg indicates average pooling
        Return : a numpy.ndarray containing the convolved images
    '''
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
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
            conv[:, i, j, :] = pooling(images[:, x:x+kh, y:y+kw, :],
                                       axis=(1, 2))
    return conv
