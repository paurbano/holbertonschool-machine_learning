#!/usr/bin/env python3
'''Strided Convolution'''

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    '''performs a valid convolution on grayscale images
    Args:
        images:array with shape (m, h, w) containing multiple grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel: array with shape (kh, kw) containing the kernel
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
        Return : a numpy.ndarray containing the convolved images
    '''
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]
    if type(padding) is tuple:
        # padding
        padh = padding[0]
        padw = padding[1]
    elif padding == 'same':
        padh = (((h - 1) * sh - h + kh) / 2) + 1
        padw = (((w - 1) * sw - w + kw) / 2) + 1
    elif padding == 'valid':
        # calculate padding
        padh = padw = 0

    # new dimensions with stride
    nh = int(((h + (2 * padh) - kh) / sh)) + 1
    nw = int(((w + (2 * padw) - kw) / sw)) + 1
    # output
    convolved = np.zeros([m, nh, nw])
    # pad images
    pad = ((0, 0), (padh, padh), (padw, padw))
    imagepaded = np.pad(images, pad_width=pad, mode='constant',
                        constant_values=0)
    # Loop over every pixel of the output
    for i in range(nh):
        for j in range(nw):
            x = i * sh
            y = j * sw
            # slice every image according to kernel size
            image = imagepaded[:, x: x+kh, y: y+kw]
            # element-wise multiplication of the kernel and the image
            convolved[:, i, j] = np.multiply(image, kernel).sum(axis=(1, 2))
    return convolved
