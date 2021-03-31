#!/usr/bin/env python3
'''performs a random crop of an image'''
import tensorflow as tf


def crop_image(image, size):
    '''performs a random crop of an image
    Args:
        image is a 3D tf.Tensor containing the image to crop
        size is a tuple containing the size of the crop
    Returns the cropped image
    '''
    seed = np.random.randint(1234)
    x = tf.random_crop(image, size=size, seed=seed)
    return x
