#!/usr/bin/env python3
'''convolution on grayscale images'''

import numpy as np


def convolve_grayscale_same(images, kernel):
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
    # calculate padding
    padh = int(kh / 2)
    padw = int(kw / 2)
    pad = ((0, 0), (padh, padh), (padw, padw))
    convolved = np.zeros([m, h, w])
    # pad images
    imagepaded = np.pad(images, pad_width=pad, mode='constant',
                        constant_values=0)
    # Loop over every pixel of the output
    for x in range(h):
        for y in range(w):
            # slice every image according to kernel size
            image = imagepaded[:, x:x+kh, y:y+kw]
            # element-wise multiplication of the kernel and the image
            convolved[:, x, y] = np.multiply(image, kernel).sum(axis=(1, 2))
    # Making sure output shape is correct
    # assert(convolved.shape == (m, n_h, n_w))
    return convolved
