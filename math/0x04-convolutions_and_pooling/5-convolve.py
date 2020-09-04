#!/usr/bin/env python3
'''Multiple Kernels'''

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    '''performs a valid convolution on grayscale images
    Args:
        images:array shape (m, h, w, c) containing multiple grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image
        kernels: array with shape (kh, kw, c, nc) containing the kernel
            kh is the height of the kernel
            kw is the width of the kernel
            nc is the number of kernels
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
    c = kernels.shape[2]
    kh = kernels.shape[0]
    kw = kernels.shape[1]
    sh = stride[0]
    sw = stride[1]
    nc = kernels.shape[3]
    # # calculate padding according to type
    if type(padding) is tuple:
        # padding
        padh = padding[0]
        padw = padding[1]
    elif padding == 'same':
        padh = int((((h - 1) * sh - h + kh) / 2)) + 1
        padw = int((((w - 1) * sw - w + kw) / 2)) + 1
    elif padding == 'valid':
        padh = padw = 0

    # new dimensions with stride
    nh = int(((h + (2 * padh) - kh) / sh)) + 1
    nw = int(((w + (2 * padw) - kw) / sw)) + 1
    # output
    # add amount of kernel
    conv = np.zeros((m, nh, nw, nc))
    # pad images add dimension at the for channels
    pad = ((0, 0), (padh, padh), (padw, padw), (0, 0))
    imagepaded = np.pad(images, pad_width=pad, mode='constant',
                        constant_values=0)
    # Loop over every pixel of the output
    for i in range(nh):
        for j in range(nw):
            for k in range(nc):
                # apply strided
                x = i * sh
                y = j * sw
                # slice every image according to kernel size
                # add dimension at the end for channels
                image = imagepaded[:, x: x+kh, y: y+kw, :]
                # element-wise multiplication of the kernel and the image
                conv[:, i, j, k] = np.multiply(image, kernels[:, :, :, k]).\
                    sum(axis=(1, 2, 3))
    return conv
