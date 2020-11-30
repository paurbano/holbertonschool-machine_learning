#!/usr/bin/env python3
''' "Vanilla" Autoencoder
https://blog.keras.io/building-autoencoders-in-keras.html
'''
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    '''creates an autoencoder:
    Args:
        input_dims: is an integer containing the dimensions of the model input
        hidden_layers: is a list containing the number of nodes for each hidden
                        layer in the encoder, respectively
            the hidden layers should be reversed for the decoder
        latent_dims: is an integer containing the dimensions of the latent
                    space representation
        lambtha: is the regularization parameter used for L1 regularization on
                the encoded output
        Returns: encoder, decoder, auto
            encoder: is the encoder model
            decoder: is the decoder model
            auto: is the full autoencoder model
        The autoencoder model should be compiled using adam optimization and
        binary cross-entropy loss
    '''
    # define the input to the encoder
    inputs = keras.Input(shape=(input_dims,))

    # "encoded" is the encoded representation of the input
    # Add a L1 activity regularizer
    encoded = keras.layers.Dense(hidden_layers[0], activation='relu')(inputs)
    # loop over the number of nodes for each hidden layer
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoded)

    regularizer = keras.regularizers.l1(lambtha)
    # latent space
    encoded = keras.layers.Dense(latent_dims, activation='relu',
                                 activity_regularizer=regularizer)(encoded)

    # build the encoder model
    encoder = keras.Model(inputs, encoded)

    # start building the decoder model which will accept the
    # output of the encoder as its inputs
    # "decoded" is the lossy reconstruction of the input
    latentInputs = keras.Input(shape=(latent_dims,))
    decoded = latentInputs

    # loop over our number of filters again, but this time in
    # reverse order
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoded)

    # this is the last layer before output,
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    # Create the decoder model
    decoder = keras.Model(latentInputs, decoded)

    # This model maps an input to its reconstruction
    # autoencoder is the encoder + decoder
    autoencoder = keras.Model(inputs, decoder(encoder(inputs)))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, autoencoder
