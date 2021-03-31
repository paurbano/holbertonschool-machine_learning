#!/usr/bin/env python3
''' performs PCA color augmentation
https://aparico.github.io/
'''

import numpy as np
import tensorflow as tf


def pca_color(image, alphas):
    '''performs PCA color augmentation as described 
    Args:
        image is a 3D tf.Tensor containing the image to change
        alphas a tuple of length 3 containing the amount that each channel
                should change
    Returns the augmented image
    '''
    # Step 1. Load the image(s) as a numpy array with (h, w, rgb) shape as
    # integers between 0 to 255
    img_to_array = tf.keras.preprocessing.image.img_to_array(image)
    orig_img = tf.keras.preprocessing.image.img_to_array(image)
    # Step 2. Convert the range of pixel values from 0-255 to 0-1
    img_to_array = img_to_array / 255.0
    # Step 3. Flatten the image to columns of RGB (3 columns)
    img_rs = img_to_array.reshape(-1, 3)
    # Step 4. Centering the pixels around their mean
    img_centered = img_rs - np.mean(img_rs, axis=0)
    # Step 5. Calculate the 3x3 covariance matrix using numpy.cov.
    # The parameter rowvar is set as False because each column represents a 
    # variable, while rows contain the values.
    img_cov = np.cov(img_centered, rowvar=False)
    # Step 6. Calculate the eigenvalues (3x1 matrix) and eigenvectors
    # (3x3 matrix) of the 3 x3 covariance matrix using numpy.linalg.eigh
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)
    # Then, sort the eigenvalues and eigenvectors
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]
    # you will finally get eigenvector matrix [p1, p2, p3] as:
    m1 = np.column_stack((eig_vecs))
    # Step 7. Get a 3x1 matrix of eigenvalues multipled by a random variable
    # drawn from a Gaussian distribution with mean=0 and sd=0.1
    # using numpy.random.normal
    m2 = np.zeros((3, 1))
    alpha = np.random.normal(0, 0.1)
    # Step 8. Create and add the vector (add_vect) that we're going to add to
    # each pixel
    m2[:, 0] = alphas * eig_vals[:]
    add_vect = np.matrix(m1) * np.matrix(m2)
    # RGB
    for idx in range(3):
        orig_img[..., idx] += add_vect[idx]

    # Step 9. Convert the range of arrays from 0-1 to 0-255 (u-int8)
    orig_img = np.clip(orig_img, 0.0, 255.0)
    orig_img = orig_img.astype(np.uint8)
    # Step 10. Convert the array of the augmented image back to jpg using
    # Image.fromarray
    # img = tf.keras.preprocessing.image.array_to_img(orig_img)
    return orig_img
