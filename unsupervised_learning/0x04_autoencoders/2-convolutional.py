#!/usr/bin/env python3
'''Convolutional Autoencoder
- https://blog.keras.io/building-autoencoders-in-keras.html
- https://idiotdeveloper.com/building-convolutional-autoencoder-
using-tensorflow-2/
- https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-
tensorflow-and-deep-learning/
'''
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    '''creates a convolutional autoencoder
    Args:
        input_dims: is an integer containing the dimensions of the model input
        filters: is a list containing the number of filters for each
                convolutional layer in the encoder, respectively
            the filters should be reversed for the decoder
        latent_dims: is a tuple of integers containing the dimensions of the
                    latent space representation
        Returns: encoder, decoder, auto
            encoder: is the encoder model
            decoder: is the decoder model
            auto: is the full autoencoder model
        The autoencoder model should be compiled using adam optimization and
        binary cross-entropy loss
    '''
    # define the input for the model
    inputs = keras.Input(shape=(input_dims))
    encoded = inputs

    # loop over the number of filters
    for f in filters:
        encoded = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                      padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)

    # build the encoder model
    # encoder = keras.Model(inputs, latent)
    encoder = keras.Model(inputs, encoded)

    # start building the decoder model which will accept the
    # output of the encoder as its inputs
    latentInputs = keras.Input(shape=(latent_dims))

    # loop over our number of filters again, but this time in
    # reverse order
    x = latentInputs
    fr = filters.copy()
    fr.reverse()
    for f in range(len(fr)):
        if (f == len(fr) - 1):
            x = keras.layers.Conv2D(filters[f], (3, 3), activation='relu',
                                    padding='valid')(x)
        else:
            x = keras.layers.Conv2D(filters[f], (3, 3), activation='relu',
                                    padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    # build the decoder model
    decoded = keras.layers.Conv2D(input_dims[2], (3, 3), activation='sigmoid',
                                  padding='same')(x)
    decoder = keras.Model(latentInputs, decoded)

    # autoencoder encoder + decoder
    autoencoder = keras.Model(inputs, decoder(encoder(inputs)))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
