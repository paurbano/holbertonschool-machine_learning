#!/usr/bin/env python3
'''randomly shears an image
https://stackoverflow.com/questions/65545653/
apply-random-shear-augment-to-image-tensor
'''


import tensorflow as tf


def shear_image(image, intensity):
    '''randomly shears an image
    Args:
        image is a 3D tf.Tensor containing the image to shear
        intensity is the intensity with which the image should be sheared
    Returns the sheared image
    '''
    array_inputs = tf.keras.preprocessing.image.img_to_array(image)
    sheared = tf.keras.preprocessing.image.random_shear(array_inputs,
                                                        intensity,
                                                        row_axis=0,
                                                        col_axis=1,
                                                        channel_axis=2)
    img_shared = tf.keras.preprocessing.image.array_to_img(sheared)
    return img_shared
