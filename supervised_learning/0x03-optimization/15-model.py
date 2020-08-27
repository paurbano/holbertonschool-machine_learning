#!/usr/bin/env python3
'''Put it all together'''


import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    '''returns two placeholders, x and y
        nx: the number of feature columns in our data
        classes: the number of classes in our classifier
        return: placeholders named x and y
            x is the placeholder for the input data
            y is the placeholder for the one-hot labels
    '''
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y


def create_layer(prev, n, activation):
    '''create a layer
        prev: tensor output of the previous layer
        n: number of nodes in the layer to create
        activation: activation function that the layer should use
        Returns: the tensor output of the layer
    '''
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    '''y = tf.layers.dense(prev, units=n, activation=activation,
            kernel_initializer=kerinit, name='layer')'''
    y = tf.layers.Dense(n, activation, kernel_initializer=init, name='layer')
    return y(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    '''creates the forward propagation graph for the neural network
        x: placeholder for the input data
        layer_sizes: list with the number of nodes in each layer
                     of the network
        activations: list with the activation functions for each layer
                     of the network
        Returns: the prediction of the network in tensor form
    '''
    # if number of layers are equal to activation functions
    if len(layer_sizes) == len(activations):
        # prediction for first layer
        Z = create_layer(x, layer_sizes[0], activations[0])
        # that is the input for next and so on...
        for i in range(1, len(layer_sizes)):
            Z = create_layer(Z, layer_sizes[i], activations[i])
        return Z
    return None


def calculate_accuracy(y, y_pred):
    ''' calculates the accuracy of a prediction:
        @y: placeholder for the labels of the input data
        @y_pred: tensor containing the network’s predictions
        Returns: a tensor containing the decimal accuracy of the prediction
    '''
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''creates a learning rate decay operation in tensorflow
      using inverse time decay

        @alpha: is the original learning rate
    @decay_rate: weight used to determine the rate at which alpha will decay
        @global_step: number of passes of gradient descent that have elapsed
        @decay_step: number of passes of gradient descent that should occur
                    before alpha is decayed further
        Returns: the learning rate decay operation
    '''
    lrd = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                      decay_rate, staircase=True)
    return lrd


def Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''training operation for a neural network using
        the Adam optimization algorithm

        @loss: is the loss of the network
        @alpha: is the learning rate
        @beta1: is the weight used for the first moment
        @beta2: is the weight used for the second moment
        @epsilon: is a small number to avoid division by zero
        Returns: the Adam optimization operation
    '''
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    adam = optimizer.minimize(loss)
    return adam


def shuffle_data(X, Y):
    '''shuffles the data points in two matrices the same way:
       X is the first numpy.ndarray of shape (m, nx) to shuffle
            m is the number of data points
            nx is the number of features in X
        Y is the second numpy.ndarray of shape (m, ny) to shuffle
            m is the same number of data points as in X
            ny is the number of features in Y
        Returns: the shuffled X and Y matrices
    '''
    shuffler = np.random.permutation(len(X))
    return X[shuffler], Y[shuffler]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    '''
    builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay, and
    batch normalization

    @Data_train: tuple containing the training inputs and training labels,
                respectively
    @Data_valid: is a tuple containing the validation inputs and validation
                labels, respectively
    @layers: list containing the number of nodes in each layer of the network
    @activation: list containing the activation functions used for each layer
                of the network
    @alpha: is the learning rate
    @beta1: is the weight for the first moment of Adam Optimization
    @beta2: is the weight for the second moment of Adam Optimization
    @epsilon: is a small number used to avoid division by zero
    @decay_rate: decay rate for inverse time decay of the learning rate
                (the corresponding decay step should be 1)
    @batch_size: number of data points that should be in a mini-batch
    @epochs: number of times the training should pass through the whole dataset
    @save_path: is the path where the model should be saved to
    Returns: the path where the model was saved
    '''
    # create placeholders
    x, y = create_placeholders(Data_train[0].shape[1], Data_train[1].shape[1])
    # forward propagation
    y_pred = forward_prop(x, layers, activations)
    # accuracy
    accuracy = calculate_accuracy(y, y_pred)
    # loss with softmax entropy
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    # learning rate decay
    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign(global_step, global_step + 1)
    alpha_tr = learning_rate_decay(alpha, decay_rate, global_step, 1)

    # train
    train_op = Adam_op(loss, alpha_tr, beta1, beta2, epsilon)

    # add to graph’s collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)
    # Create a saver.
    saver = tf.train.Saver()
    # initialize variables
    init = tf.global_variables_initializer()

    # define number of steps
    print(Data_train[0].shape[0])
    steps = round(Data_train[0].shape[0] / batch_size)
    length = Data_train[0].shape[0]

    with tf.Session() as session:
        session.run(init)
        for epoch in range(epochs + 1):
            # ={x: X_valid, y: Y_valid}
            feed_dict = {x: Data_train[0], y: Data_train[1]}
            # train values
            t_accur = session.run(accuracy, feed_dict)
            t_loss = session.run(loss, feed_dict)
            # valid values
            vaccur = session.run(accuracy, feed_dict={x: Data_valid[0],
                                                      y: Data_valid[1]})
            v_loss = session.run(loss, feed_dict={x: Data_valid[0],
                                                  y: Data_valid[1]})
            print('After {} epochs:'.format(epoch))
            print('\tTraining Cost: {}'.format(t_loss))
            print('\tTraining Accuracy: {}'.format(t_accur))
            print('\tValidation Cost: {}'.format(v_loss))
            print('\tValidation Accuracy: {}'.format(vaccur))
            if epoch != epochs:
                # pointer where
                start = 0
                end = batch_size
                # shuffle training data before each epoch
                X_trainS, Y_trainS = shuffle_data(Data_train[0], Data_train[1])
                for step in range(1, steps + 2):
                    # slice train data according to mini bach size every
                    train_batch = X_trainS[start:end]
                    train_label = Y_trainS[start:end]
                    feed_dict = {x: train_batch, y: train_label}
                    # run train operation
                    b_train = session.run(train_op, feed_dict)
                    if step % 100 == 0:
                        # compute cost and accuracy every 100 steps
                        b_cost = session.run(loss, feed_dict)
                        b_accuracy = session.run(accuracy, feed_dict)
                        print('\tStep {}:'.format(step))
                        print('\t\tCost: {}'.format(b_cost))
                        print('\t\tAccuracy: {}'.format(b_accuracy))
                    # increment point to slice according to batch size
                    start = start + batch_size
                    if (length - start) < batch_size:
                        end = end + (length - start)
                    else:
                        end = end + batch_size
            # increment global_step for learning decay and train ops
            session.run(increment_global_step)
        return saver.save(session, save_path)
