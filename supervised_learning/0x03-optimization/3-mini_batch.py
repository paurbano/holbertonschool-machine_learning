#!/usr/bin/env python3
'''using mini-batch gradient descent'''

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    '''trains a loaded neural network model using mini-batch gradient descent
        @X_train: array of shape (m, 784) containing the training data
            m: is the number of data points
            784: is the number of input features
        @Y_train: one-hot array of shape (m, 10) containing the training labels
            10 is the number of classes the model should classify
        X_valid: array of shape (m, 784) containing the validation data
        Y_valid: array of shape (m, 10) containing the validation labels
        batch_size: is the number of data points in a batch
        epochs: number of times the training should pass through the
                whole dataset
        load_path: is the path from which to load the model
        save_path: path to where the model should be saved after training
        Returns: the path where the model was saved
    '''
    with tf.Session() as session:
        # load meta graph and restore weights
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(session, load_path)
        # access placeholders and ops
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        # define number of steps
        # print(batch_size)
        steps = round(len(X_train) / batch_size)
        # print(steps)
        # print(X_train.shape[0])
        length = X_train.shape[0]
        for epoch in range(epochs + 1):
            # ={x: X_valid, y: Y_valid}
            feed_dict = {x: X_train, y: Y_train}
            # train values
            t_accur = session.run(accuracy, feed_dict)
            t_loss = session.run(loss, feed_dict)
            # valid values
            # feed_dict = {x: X_valid, y: Y_valid}
            vaccur = session.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            v_loss = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
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
                X_trainS, Y_trainS = shuffle_data(X_train, Y_train)
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
        # print('after {} steps - start:{} - end:{}'.format(step,start,end))
        return saver.save(session, save_path)
