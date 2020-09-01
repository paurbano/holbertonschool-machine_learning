#!/usr/bin/env python3
'''trains a model using mini-batch gradient descent'''


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    '''trains a model using mini-batch gradient descent
    Args:
        network is the model to train
        data is a numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing the
                labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        validation_data: is the data to validate the model with, if not None
        early_stopping: boolean that indicates whether early stopping should be used
                early stopping should only be performed if validation_data exists
                early stopping should be based on validation loss
        patience: is the patience used for early stopping
        verbose: boolean that determines if output should be printed during
                training
        shuffle: boolean that determines whether to shuffle the batches every
                epoch. Normally, it is a good idea to shuffle, but for
                reproducibility, we have chosen to set the default to False.
        Returns: the History object generated after training the model
    '''
    if early_stopping and validation_data is not None:
        callback = [K.callbacks.EarlyStopping(monitor='loss', patience=patience)]
        History= network.fit(x=data,y=labels, batch_size=batch_size, epochs=epochs,
                        verbose=verbose, validation_data=validation_data,
                        shuffle=shuffle, callbacks=callback)
    else:
        # print('sin validation data')
        History= network.fit(x=data,y=labels, batch_size=batch_size, epochs=epochs,
                        verbose=verbose, validation_data=validation_data,
                        shuffle=shuffle)
    return History
