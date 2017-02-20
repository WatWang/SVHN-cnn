import tensorflow as tf


def conv2d(x, n_output, filter_size=3, strides=1, padding='SAME', activation=tf.nn.relu, name='conv2d', reuse=None):
    '''
    :param x: Input data
    :param n_output: Number of convolution layer output
    :param filter_size: Convolution kernel size
    :param strides: Parameter of strides
    :param padding: 'SAME' or 'VALID'
    :param activation: Set the non-linear function
    :return: f(conv(W,x)+b)
    '''

    with tf.variable_scope(name or 'conv2d', reuse=reuse):
        # Set parameter W & b
        W = tf.get_variable(shape=[filter_size, filter_size, x.get_shape()[-1], n_output], initializer=tf.contrib.layers.xavier_initializer_conv2d(),name='W')
        b = tf.get_variable(shape=[n_output], initializer=tf.constant_initializer(0.0),name='b')

        # Apply convolution
        conv = tf.nn.conv2d(input=x, filter=W, strides=[1, strides, strides, 1], padding=padding)

        # Calculate the output: output = conv(W,x) + b
        h = tf.nn.bias_add(conv, b)

        # Apply non_linear
        if activation:
            h = activation(h)

    return h

def max_pooling(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pooling', reuse=None):
    '''
    :param x: Input data
    :param ksize: Size of window for each dimension
    :param strides: Parameter of strides
    :param padding: 'SAME' or 'VALID'
    :return: Pooling result
    '''

    # Apply pooling
    h = tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)

    return h

def linear(x, n_output, activation=tf.nn.relu, name = "linear",reuse=None):
    '''
    :param x: Input data
    :param n_output: Output length
    :param activation: Set the non-linear function
    :return: f(W*x+b)
    '''

    with tf.variable_scope(name or "linear", reuse=reuse):
        # Set parameter W & b
        W = tf.get_variable(shape=[x.get_shape()[1], n_output], initializer=tf.contrib.layers.xavier_initializer(), name='W')
        b = tf.get_variable(shape=[n_output], initializer=tf.constant_initializer(0.0), name='b')

        # Apply linear predict
        h = tf.nn.bias_add(tf.matmul(x, W), b)

        # Apply non_linear
        if activation:
            h = activation(h)

    return h