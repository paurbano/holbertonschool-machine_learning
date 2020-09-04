#!/usr/bin/env python3
'''convolution on grayscale images'''

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    '''performs a valid convolution on grayscale images
    Args:
        images:array with shape (m, h, w) containing multiple grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel: array with shape (kh, kw) containing the kernel
            kh is the height of the kernel
            kw is the width of the kernel
        padding is a tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
            the image should be padded with 0’s
        Return : a numpy.ndarray containing the convolved images
    '''
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    # padding
    padh = padding[0]
    padw = padding[1]
    pad = ((0, 0), (padh, padh), (padw, padw))
    # new dimensions
    nh = h + (2 * padh) - kh + 1
    nw = w + (2 * padw) - kw + 1
    convolved = np.zeros([m, nh, nw])
    # pad images
    imagepaded = np.pad(images, pad_width=pad, mode='constant',
                        constant_values=0)
    # Loop over every pixel of the output
    for x in range(nh):
        for y in range(nw):
            # slice every image according to kernel size
            image = imagepaded[:, x:x+kh, y:y+kw]
            # element-wise multiplication of the kernel and the image
            convolved[:, x, y] = np.multiply(image, kernel).sum(axis=(1, 2))
    # Making sure output shape is correct
    # assert(convolved.shape == (m, n_h, n_w))
    return convolved
