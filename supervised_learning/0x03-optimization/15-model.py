#!/usr/bin/env python3
'''Put it all together'''


import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data
Adam_op = __import__('10-Adam').create_Adam_op


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
    with tf.Session() as session:
        # create placeholders
        x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

        # forward propagation
        # y_pred = forward_prop(x, layer_sizes, activations)
        # prediction for first layer
        Z = create_layer(x, layer_sizes[0], activations[0])
        # that is the input for next and so on...
        for i in range(1, len(layer_sizes)):
            Z = create_layer(Z, layer_sizes[i], activations[i])

        # accuracy
        accuracy = calculate_accuracy(y, y_pred)
        # loss
        loss = calculate_loss(y, y_pred)
        # train
        train_op = Adam_op(alpha, beta1, beta2, epsilon)
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
        steps = round(len(X_train) / batch_size)
        length = X_train.shape[0]
        for epoch in range(epochs + 1):
            # ={x: X_valid, y: Y_valid}
            feed_dict = {x: X_train, y: Y_train}
            # train values
            t_accur = session.run(accuracy, feed_dict)
            t_loss = session.run(loss, feed_dict)
            # valid values
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
