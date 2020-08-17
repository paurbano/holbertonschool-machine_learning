#!/usr/bin/env python3
'''train neural netowrk classifier'''


import tensorflow as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    '''builds, trains, and saves a neural network classifier
        @X_train: numpy.ndarray containing the training input data
        @Y_train: numpy.ndarray containing the training labels
        @X_valid: numpy.ndarray containing the validation input data
        @Y_valid: numpy.ndarray containing the validation labels
        @layer_sizes: list number of nodes in each layer of the network
        @activations: list activation functions for each layer of the network
        @alpha: learning rate
        @iteratios: number of iterations to train over
        @save_path :where to save the model
    '''
    # reset the graph
    tf.reset_default_graph()
    # create placeholders
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    # forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)
    # accuracy
    accuracy = calculate_accuracy(y, y_pred)
    # loss
    loss = calculate_loss(y, y_pred)
    # train
    train_op = create_train_op(loss, alpha)
    # add to graph’s collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)
    # Create a saver.
    saver = tf.train.Saver()
    # initialize variables
    init = tf.global_variables_initializer()
    # create new session
    with tf.Session() as sess:
        sess.run(init)
        # train over number of iterations to train over
        for i in range(iterations + 1):
            # train values
            t_accur = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            t_loss = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            # valid values
            v_accur = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            v_loss = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations:
                print('After {} iterations:'.format(i))
                print('\tTraining Cost: {}'.format(t_loss))
                print('\tTraining Accuracy: {}'.format(t_accur))
                print('\tValidation Cost: {}'.format(v_loss))
                print('\tValidation Accuracy: {}'.format(v_accur))
            if i != iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        # save the model in a file
        return saver.save(sess, save_path)
