from import_data_with_color import importdata
from network_framework_auto import framework
import openpyxl
import numpy as np
import tensorflow as tf
import time


def select_data(feature_train, label_train, times, batch_size, num_of_train):
    if (times*100) % num_of_train < (times*100 + batch_size-1)%num_of_train:
        return (feature_train[range((times*100) % num_of_train, (times*100+batch_size-1) % num_of_train)],
                label_train[range((times*100) % num_of_train, (times*100+batch_size-1)%num_of_train)])
    return (np.append(feature_train[range(0, (times*100+batch_size-1)%num_of_train)],
                      feature_train[range((times*100)%num_of_train, num_of_train)],axis=0),
            np.append(label_train[range(0, (times * 100 + batch_size - 1) % num_of_train)],
                      label_train[range((times * 100) % num_of_train, num_of_train)],axis=0))


def nn_train(train_option,framework_file):
    # Switch if load model and if save model
    pre_trained = train_option['pre_trained']
    save_model = train_option['save_model']

    # Set parameters
    train_batch_size = train_option['batch_size']
    range_times = train_option['range_times']
    data_name = train_option['data_name']
    num_class = train_option['num_class']
    learning_rate = train_option['learning_rate']
    keep_prob = train_option['keep_prob']
    sleep_time = train_option['sleep_time']
    optimize_option = train_option['optimize_option']
    already_train_times = 0

    # Set model path
    if save_model:
        model_save_name = train_option['model_save_name']
        model_save_path = './' + train_option['model_path'] + '/' + model_save_name

    # Protect if no model find
    if not train_option['model_load_name']:
        pre_trained = False

    # Set load model path
    if pre_trained:
        model_load_name = train_option['model_load_name']
        model_load_path = './' + train_option['model_path'] + '/' + model_load_name
        already_train_times = train_option['already_train_times']

    # Set tensorboard direction
    if train_option['tensorboard']:
        board_path = './' + train_option['tensorboard']

    # Import train data
    (feature_train, label_train) = importdata(data_name)

    # Build up network framework
    model = framework(feature_train.shape,num_class,framework_data=framework_file,keep_prob=keep_prob)

    # Set cost and train step
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.y_pred, labels=model.y))
    if optimize_option == 'AdamOptimizer' or optimize_option == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    if optimize_option == 'GradientDescentOptimizer' or optimize_option == 'Grad':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    if optimize_option == 'FtrlOptimizer' or optimize_option == 'Ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(cost)
    if optimize_option == 'MomentumOptimizer' or optimize_option == 'Mome':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate).minimize(cost)
    if optimize_option == 'AdagradOptimizer' or optimize_option == 'Adag':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
    if optimize_option == 'AdadeltaOptimizer' or optimize_option == 'Adad':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

    if save_model:
        # Initialize saver
        saver = tf.train.Saver()

    # Create optimizer and session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    if train_option['tensorboard']:
        # sess.graph contains the graph definition; that enables the Graph Visualizer
        file_writer = tf.train.SummaryWriter(board_path, sess.graph)
        tf.summary.scalar('loss', cost)
        merged_summary_op = tf.summary.merge_all()

    # Load per-trained model
    if pre_trained:
        saver.restore(sess, model_load_path)

    # Record batch time
    time_batch = time.clock()

    # Train
    for epoch in range(range_times):
        avg_cost = 0.
        total_batch = int(feature_train.shape[0] / train_batch_size)

        # Record time
        time_range = time.clock()

        # Loop over all batches
        for i in range(total_batch):
            # Apply train
            batch_feature = feature_train[train_batch_size * i:min(train_batch_size * (i + 1) - 1, feature_train.shape[0])]
            batch_label = label_train[train_batch_size * i:min(train_batch_size * (i + 1) - 1, label_train.shape[0])]

            # Run optimization op (backprop) and cost op (to get loss value)
            if train_option['tensorboard']:
                _, c, merged_summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={model.x: batch_feature, model.y: batch_label})
            else:
                _, c = sess.run([optimizer, cost], feed_dict={model.x: batch_feature, model.y: batch_label})

            # Compute average loss
            avg_cost += c / total_batch

            # Display train percent
            if i % int(total_batch/100) == 0:
                print('.', end='')

            # Wait for cooling gpu
            if sleep_time and (i % int(1000/train_batch_size + 1) ) == 0:
                # Collect batch time
                time_batch = time.clock() - time_batch

                # Sleep
                time.sleep(time_batch * sleep_time)

                # Record batch time
                time_batch = time.clock()

        # Display logs per epoch step
        print('Train finished')
        time_range = time.clock() - time_range
        if epoch % 1 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost), "Use:", '%02d' % time_range,"seconds")

            # Set record files name
            record_file_name = './' + train_option['record_path'] + '/' + model_save_name + '-' + str(already_train_times + epoch) + '.xlsx'

            # Open file
            workbook = openpyxl.load_workbook(filename=record_file_name)

            # Create work sheet
            worksheet = workbook.create_sheet(title='train')

            # Input records
            worksheet['A1'] = 'Train data'
            worksheet['B1'] = data_name
            worksheet['A2'] = 'Train times'
            worksheet['B2'] = str(already_train_times + epoch + 1)
            worksheet['A3'] = 'Optimizer type'
            worksheet['B3'] = optimize_option
            worksheet['A3'] = 'Loss'
            worksheet['B3'] = '%04f' % avg_cost
            worksheet['A4'] = 'Use time'
            worksheet['B4'] = '%03f' % time_range


            # Save record
            workbook.save(filename=record_file_name)

        if save_model:
            # Save network model
            save_path = saver.save(sess, model_save_path, global_step=(already_train_times + epoch))
            print("Model saved in file: %s" % save_path)

        if train_option['tensorboard']:
            # Save tensor board
            file_writer.add_summary(merged_summary, epoch)

    print("Optimization Finished!")
    sess.close()

    return True

