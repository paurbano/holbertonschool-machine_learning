#!/usr/bin/env python3
'''save the model'''


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    '''save the best iteration of the model
    Args:
        network is the model to train
        data is a numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing the
                labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        validation_data: is the data to validate the model with, if not None
        early_stopping: boolean indicates whether early stopping should be used
            early stopping: should only be performed if validation_data exists
            early stopping: should be based on validation loss
        patience: is the patience used for early stopping
        learning_rate_decay: is a boolean that indicates whether learning rate
                             decay should be used
            * should only be performed if validation_data exists
            * the decay should be performed using inverse time decay
            * should decay in a stepwise fashion after each epoch
            *each time the learning rate updates, Keras should print a message
        alpha: is the initial learning rate
        decay_rate: is the decay rate
        verbose: boolean that determines if output should be printed during
                training
        shuffle: boolean that determines whether to shuffle the batches every
                epoch. Normally, it is a good idea to shuffle, but for
                reproducibility, we have chosen to set the default to False.
        Returns: the History object generated after training the model
    '''
    callback = []
    if early_stopping and validation_data:
        callback = [K.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=patience)]

    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            ''' This function keeps the initial learning rate for the
                first ten epochs and decreases it exponentially after that
            '''
            return alpha / (1 + decay_rate * epoch)
        callback.append(K.callbacks.LearningRateScheduler(scheduler,
                                                          verbose=1))
        '''
        History = network.fit(x=data, y=labels, batch_size=batch_size,
                              verbose=verbose, epochs=epochs, shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=callback)
        '''
    if filepath:
        checkpoint = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                 monitor='val_loss',
                                                 save_best_only=save_best,
                                                 mode='auto')
        callback.append(checkpoint)
    '''
    else:
        History = network.fit(x=data, y=labels, batch_size=batch_size,
                              epochs=epochs, validation_data=validation_data,
                              verbose=verbose, shuffle=shuffle)
    '''
    History = network.fit(x=data, y=labels, batch_size=batch_size,
                          verbose=verbose, epochs=epochs, shuffle=shuffle,
                          validation_data=validation_data, callbacks=callback)
    return History
