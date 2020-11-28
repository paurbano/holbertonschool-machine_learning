#!/usr/bin/env python3
''' Variational Autoencoder
https://blog.keras.io/building-autoencoders-in-keras.html
https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-\
autoencoder-with-keras/
'''
import tensorflow.keras as keras


def sampling(args):
    '''sample new similar points from the latent space
    '''
    z_mean, z_log_sigma, latent_dims = args
    epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], latent_dims),
                              mean=0., stddev=0.1)
    return z_mean + keras.backend.exp(z_log_sigma) * epsilon

def autoencoder(input_dims, hidden_layers, latent_dims):
    '''creates an autoencoder:
    Args:
        input_dims: is an integer containing the dimensions of the model input
        hidden_layers: is a list containing the number of nodes for each hidden
                        layer in the encoder, respectively
            the hidden layers should be reversed for the decoder
        latent_dims: is an integer containing the dimensions of the latent
                    space representation
        Returns: encoder, decoder, auto
            encoder: is the encoder model, which should output the latent
                   representation, the mean, and the log variance, respectively
            decoder: is the decoder model
            auto: is the full autoencoder model
        The autoencoder model should be compiled using adam optimization and
        binary cross-entropy loss
    '''
    inputs = keras.Input(shape=(input_dims,))
    h = keras.layers.Dense(hidden_layers[0], activation='relu')(inputs)
    z_mean = keras.layers.Dense(latent_dims)(h)
    z_log_sigma = keras.layers.Dense(latent_dims)(h)
    z = keras.layers.Lambda(sampling,output_shape=(latent_dims,))([z_mean, z_log_sigma,latent_dims])
    # Create encoder
    print('antes del encoder model')
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    # Create decoder
    print('iniciando decoder')
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = keras.layers.Dense(hidden_layers[0], activation='relu')(latent_inputs)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    print('antes del decoder model')
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    print('antes del autoencoder model')
    autoencoder = keras.Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, autoencoder
