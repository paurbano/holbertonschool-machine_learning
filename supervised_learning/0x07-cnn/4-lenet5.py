#!/usr/bin/env python3
'''modified version of the LeNet-5 architecture using tensorflow'''


import tensorflow as tf


def lenet5(x, y):
    '''modified version of the LeNet-5 architecture
    args:
        x: is a tf.placeholder of shape (m, 28, 28, 1) containing the input
           images for the network
            m is the number of images
        y: is a tf.placeholder of shape (m, 10) containing the one-hot labels
           for the network
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method:
        tf.contrib.layers.variance_scaling_initializer()
    All hidden layers requiring activation should use the relu activation
    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
            (with default hyperparameters)
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    '''
    # convolutional layer 1
    # 6 kernels 5x5, padding = same
    init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2D(
          inputs=x,
          filters=6,  # Number of filters.
          kernel_size=5,  # Size of each filter is 5x5.
          padding="same",  # padding is applied to the input.
          activation=tf.nn.relu,  # relu activation function
          kernel_initializer=init)

    # pooling layer #1
    pool1 = tf.layers.MaxPooling2D(inputs=conv1, pool_size=[2, 2],
                                   strides=(2, 2))

    # convolutional layer 2
    # 16 kernels 5x5, padding = valid
    conv2 = tf.layers.Conv2D(
          inputs=pool1,
          filters=16,  # Number of filters.
          kernel_size=5,  # Size of each filter is 5x5.
          padding="valid",  # padding is applied to the input.
          activation=tf.nn.relu,  # relu activation function
          kernel_initializer=init)

    # pooling layer #2
    pool2 = tf.layers.MaxPooling2D(inputs=conv2, pool_size=[2, 2],
                                   strides=(2, 2))

    # Reshaping output into a single dimention array for input
    # to fully connected layer
    # pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
    pool2_flat = tf.layers.Flatten()(pool2)

    # Fully connected layer #1: Has 120 neurons
    dense1 = tf.layers.Dense(inputs=pool2_flat, units=120,
                             activation=tf.nn.relu, kernel_initializer=init)

    # dense1_flat = Flatten()(dense1)
    # Fully connected layer #2: Has 84 neurons
    dense2 = tf.layers.Dense(inputs=dense1, units=84,
                             activation=tf.nn.relu, kernel_initializer=init)

    # Output layer, 10 neurons for each digit
    dense3 = tf.layers.Dense(inputs=dense2, units=10)

    softmax = tf.nn.softmax(dense3)

    # Convert our labels into one-hot-vectors
    # labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=10)

    # Compute the cross-entropy loss
    y_pred = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    # Use adam optimizer to reduce cost
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    # For testing and prediction
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return softmax, train_op, loss, accuracy
