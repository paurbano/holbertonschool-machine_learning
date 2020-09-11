#!/usr/bin/env python3
'''LeNet-5 architecture using keras'''

import tensorflow.keras as K


def lenet5(X):
    '''version of the LeNet-5 architecture using keras
    Args:
        X is a K.Input of shape (m, 28, 28, 1) containing the input images for
          the network:
            m is the number of images
        The model should consist of the following layers in order:
            Convolutional layer with 6 kernels of shape 5x5 with same padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Convolutional layer with 16 kernels of shape 5x5 with valid padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Fully connected layer with 120 nodes
            Fully connected layer with 84 nodes
            Fully connected softmax output layer with 10 nodes
        All layers requiring initialization should initialize their kernels
        with the he_normal initialization method
        All hidden layers requiring activation should use the relu activation
            function
    Returns: a K.Model compiled to use Adam optimization (with default
            hyperparameters) and accuracy metrics
    '''
    # empty model
    # model = K.Sequential()
    # C1 convolutional layer
    # initializer = K.initializers.he_normal
    conv1 = K.layers.Conv2D(6, kernel_size=(5, 5),
                            activation='relu', kernel_initializer='he_normal',
                            padding="same")(X)
    # Pool layer 1
    maxpool1 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    # C2 convolutional layer
    conv2 = K.layers.Conv2D(16, kernel_size=(5, 5),
                            activation='relu', kernel_initializer='he_normal',
                            padding='Valid')(maxpool1)
    # Pool layer 2
    maxpool2 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    # Flatten the CNN output so that we can connect it with FC layers
    flatten = K.layers.Flatten()(maxpool2)
    # FC Fully Connected Layer 120 nodes
    dense1 = K.layers.Dense(120, activation='relu',
                            kernel_initializer='he_normal')(flatten)
    # FC Fully Connected Layer 84 nodes
    dense2 = K.layers.Dense(84, activation='relu',
                            kernel_initializer='he_normal')(dense1)
    # Output Layer with softmax activation
    dense3 = K.layers.Dense(10, activation='softmax',
                            kernel_initializer='he_normal')(dense2)
    model = K.Model(inputs=X, outputs=dense3)
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(), metrics=['accuracy'])
    return model
