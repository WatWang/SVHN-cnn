import tensorflow as tf
from nn_op import  conv2d, linear

class framework(object):
    # Test a cnn for a picture classifier
    def __init__(self,shape_feature,num_class,framework_data = None,keep_prob = 1.0):
        # Set place for input and output
        self.x = tf.placeholder(tf.float32, [None, shape_feature[1], shape_feature[2], shape_feature[3]])
        self.y = tf.placeholder(tf.float32, [None, num_class])

        # If no framework data input set as an one layer linear network
        if not framework_data:
            shape = self.x.get_shape().as_list()
            layer_1 = tf.reduce_mean(self.x, axis=3)
            layer_1 = tf.reshape(layer_1, [-1, shape[1] * shape[2]])
            logits = linear(layer_1, num_class, activation=False, name="logits")

        else:
            # Initialize net data
            net_data = self.x

            # Get total layers of the network
            num_layer = len(framework_data)
            previous_layer_type = 'conv'

            for i in range(num_layer):
                layer_name = 'Layer_' + str(i + 1)

                # Load layer type
                layer_type = framework_data[layer_name]['layer_type']
                print('Layer = ', i, 'Type = ', layer_type)

                # Load output size
                output_size = framework_data[layer_name]['output_size']

                # Build linear layer
                if layer_type == 'linear':
                    # Make sure the dimension of the input is 2
                    if previous_layer_type == 'conv' or previous_layer_type == 'conv2d':

                        # Get shape
                        shape = net_data.get_shape().as_list()

                        # If it is the first layer, it can be 2d or 3d or 4d input
                        if i == 0:
                            net_data = tf.reduce_mean(net_data, axis=3)
                            net_data = tf.reshape(net_data, [-1, shape[1] * shape[2]])

                        # If not it must be from a convolution layer, so it must be 4d
                        else:
                            net_data = tf.reshape(net_data, [-1, shape[1] * shape[2] * shape[3]])

                    # Load activation function, default relu
                    try:
                        activation_in = framework_data[layer_name]['activation']
                        # Add 'tf.nn.' to input string
                        if not 'tf.nn.' in activation_in:
                            activation_in = 'tf.nn.' + activation_in

                        # Make this string to expression
                        activation = eval(activation_in)
                    except:
                        activation = tf.nn.relu

                    # Set layer name
                    current_name = layer_name

                    # Check whether it is the output layer
                    if i == num_layer - 1:
                        output_size = num_class
                        activation = False
                        current_name = 'output_layer'

                    # Apply linear
                    try:
                        net_data = linear(net_data, output_size, activation=activation, name= current_name)
                    except:
                        net_data = linear(net_data, output_size, activation=activation, name= current_name, reuse=True)

                    # Load dropout settings
                    try:
                        dropout_switch = framework_data[layer_name]['dropout_switch']
                    except:
                        dropout_switch = False
                    else:
                        # If dropout is actived, load keep_prob
                        try:
                            keep_prob_in = framework_data[layer_name]['keep_prob']
                        except:
                            keep_prob_in = keep_prob

                    # Apply dropout
                    if dropout_switch:
                        with tf.name_scope('dropout_' + layer_name):
                            net_data = tf.nn.dropout(net_data, keep_prob_in)

                if layer_type == 'conv' or layer_type == 'conv2d':
                    # For just a sub-layer the output_size can be a number
                    if not type(output_size) == list:
                        output_size = [output_size]

                    # Initialize sub_layer counter
                    conv_counter = 0

                    # Load convolution filter size, default 3
                    try:
                        conv_filter_size = framework_data[layer_name]['conv_filter_size']
                    except:
                        conv_filter_size = 3

                    # Make convolution size a list
                    if type(conv_filter_size) == int:
                        filter_size_list = [conv_filter_size]
                        for j in range(len(output_size) - 1):
                            filter_size_list.append(conv_filter_size)

                    # Load convolution strides, default 1
                    try:
                        conv_strides = framework_data[layer_name]['conv_strides']
                    except:
                        conv_strides = 1

                    # Load convolution padding, default 'SAME'
                    try:
                        conv_padding = framework_data[layer_name]['conv_padding']
                    except:
                        conv_padding = 'SAME'

                    # Turn the cycle of the convolution size list
                    for output_size_conv, filter_size in zip(output_size,filter_size_list):
                        # Count number of sub-layers
                        conv_counter += 1

                        # Set layer name
                        current_name = layer_name + '_' + str(conv_counter)

                        try:
                            # Apply convolution
                            net_data = conv2d(net_data, output_size_conv, filter_size=filter_size, strides=conv_strides,padding=conv_padding, name=current_name)
                        except:
                            net_data = conv2d(net_data, output_size_conv, filter_size=filter_size, strides=conv_strides,
                                              padding=conv_padding, name=current_name,reuse=True)

                    # Load pooling type, default max pool
                    try:
                        pool_type = framework_data[layer_name]['pool_type']
                    except:
                        pool_type = 'max_pool'

                    # Load pooling filter size, default 3
                    try:
                        pool_k_size = framework_data[layer_name]['pool_k_size']
                    except:
                        pool_k_size = [1, 2, 2, 1]

                    # Load pooling strides, default 1
                    try:
                        pool_strides = framework_data[layer_name]['pool_strides']
                    except:
                        pool_strides = [1, 2, 2, 1]

                    # Load pooling padding, default 'SAME'
                    try:
                        pool_padding = framework_data[layer_name]['pool_padding']
                    except:
                        pool_padding = 'SAME'

                    # Apply pooling
                    if pool_type == 'max_pool' or pool_type == 'MAX':
                        net_data = tf.nn.max_pool(net_data,ksize=pool_k_size,strides=pool_strides,padding=pool_padding)
                    if pool_type == 'avg_pool' or pool_type == 'AVG':
                        net_data = tf.nn.avg_pool(net_data,ksize=pool_k_size,strides=pool_strides,padding=pool_padding)

                previous_layer_type = layer_type
            # logits
            logits = net_data

        # softmax, resulting shape=[-1, n_classes]
        self.y_pred = logits
