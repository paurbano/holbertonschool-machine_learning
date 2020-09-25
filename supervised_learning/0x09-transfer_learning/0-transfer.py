#!/usr/bin/env python3
'''transfer learning'''

import tensorflow.keras as K


def preprocess_data(X, Y):
    '''pre-processes the data
    Args:
        X is a numpy.ndarray of shape (m, 32, 32, 3) containing the
        CIFAR 10 data, where m is the number of data points
        Y is a numpy.ndarray of shape (m,) containing the CIFAR 10
        labels for X
    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    '''
    X = X / 255.0
    # X = tf.image.resize(image, (200, 200))
    # # one hot encode the labels
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


if __name__ == "__main__":
    # Load the dataset:
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # preprocess data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    # load model without output layer include_top=False
    # ResNet50
    base_model = K.applications.ResNet50(weights='imagenet', include_top=False,
                                         input_shape=(200, 200, 3))

    # Inception V3
    # base_model = K.applications.InceptionV3()

    # freeze model
    base_model.trainable = False

    # transfer learning model
    # the input image of Cifar10 is 32x32 so it needs to be upscaled 3 times
    # before we can pass it through the ResNet layers
    model = K.Sequential()
    model.add(K.layers.UpSampling2D((2, 2)))
    model.add(K.layers.UpSampling2D((2, 2)))
    model.add(K.layers.UpSampling2D((2, 2)))
    model.add(base_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu'))
    # model.add(K.layers.GlobalAveragePooling2D())
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(10, activation='softmax'))

    # summarize model
    # model.summary()

    # train top layer
    # patience = 3
    model.compile(optimizer=K.optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy', metrics=['accuracy'])

    # callback = [K.callbacks.EarlyStopping(monitor='val_loss',
    #                                      patience=3)]

    model.fit(x=x_train, y=y_train, batch_size=32, epochs=5,
              validation_data=(x_test, y_test))

    model.save('cifar10.h5')
