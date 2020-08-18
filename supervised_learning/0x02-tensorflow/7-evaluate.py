#!/usr/bin/env python3
'''evaluates the output of neural network'''


import tensorflow as tf


def evaluate(X, Y, save_path):
    '''evaluates the output of a neural network:
        X: is a numpy.ndarray containing the input data to evaluate
        Y: is a numpy.ndarray containing the one-hot labels for X
        save_path: is the location to load the model from
        Returns: the network’s prediction, accuracy, and loss, respectively
    '''
    with tf.Session() as sess:
        # load meta graph and restore weights
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        # print(tf.get_collection('accuracy'))
        # graph = tf.get_default_graph()

        # access placeholders
        x = tf.get_collection('x')[0]  # graph.get_tensor_by_name("x:0")
        y = tf.get_collection('y')[0]  # graph.get_tensor_by_name("x:0")
        # graph.get_tensor_by_name("layer_2/BiasAdd:0")
        y_pred = tf.get_collection('y_pred')[0]
        # graph.get_tensor_by_name("Mean:0")
        accuracy = tf.get_collection('accuracy')[0]
        # graph.get_tensor_by_name("loss:0")
        loss = tf.get_collection('loss')[0]

        # create feed-dict to feed new data
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        accu = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss_value = sess.run(loss, feed_dict={x: X, y: Y})

        return prediction, accu, loss_value
        # return None, None, None
