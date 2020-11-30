#!/usr/bin/env python3
''' "Vanilla" Autoencoder
https://blog.keras.io/building-autoencoders-in-keras.html
https://www.machinecurve.com/index.php/2019/12/30/how-to-create-a-variational-\
autoencoder-with-keras/
'''
import tensorflow.keras as keras


def sampling(args):
    '''sample new similar points from the latent space
    '''
    z_mean, z_log_sigma = args
    batch = keras.backend.shape(z_mean)[0]
    dims = keras.backend.int_shape(z_mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dims))
    return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon


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
    # define the input to the encoder
    inputs = keras.Input(shape=(input_dims,))

    # "encoded" is the encoded representation of the input
    encoded = keras.layers.Dense(hidden_layers[0], activation='relu')(inputs)
    # loop over the number of nodes for each hidden layer
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoded)
    # mean and log variance layers
    z_mean = keras.layers.Dense(latent_dims)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims)(encoded)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

    # build the encoder model
    encoder = keras.Model(inputs, [z, z_mean, z_log_sigma])

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
    # Define loss
    def kl_reconstruction_loss(true, pred):
        # Reconstruction loss
        reconstruction_loss = keras.losses.binary_crossentropy(inputs,
                                                               vae_outputs)
        reconstruction_loss *= input_dims
        # KL divergence loss
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) -\
            keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(reconstruction_loss + kl_loss)

    vae_outputs = decoder(encoder(inputs))
    autoencoder = keras.Model(inputs, vae_outputs)

    autoencoder.compile(optimizer='adam', loss=kl_reconstruction_loss)
    return encoder, decoder, autoencoder
