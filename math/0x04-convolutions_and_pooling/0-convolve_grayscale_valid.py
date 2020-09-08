#!/usr/bin/env python3
'''convolution on grayscale images'''

import numpy as np


def convolve_grayscale_valid(images, kernel):
    '''performs a valid convolution on grayscale images
    Args:
        images:array with shape (m, h, w) containing multiple grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel: array with shape (kh, kw) containing the kernel
            kh is the height of the kernel
            kw is the width of the kernel
        Return : a numpy.ndarray containing the convolved images
    '''
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    n_h = h - kh + 1
    n_w = w - kw + 1
    # size output convolved
    convolved = np.zeros([m, n_h, n_w])
    # Loop over every pixel of the output
    for x in range(n_h):
        for y in range(n_w):
            # slice every image according to kernel size
            image = images[:, x:x+kh, y:y+kw]
            # apply convolution:
            # element-wise multiplication of the kernel and the image
            # and sum it
            convolved[:, x, y] = np.multiply(image, kernel).sum(axis=(1, 2))
    # Making sure output shape is correct
    # assert(convolved.shape == (m, n_h, n_w))
    return convolved
