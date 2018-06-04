# coding=utf-8

import numpy as np
import time
import tensorflow as tf
from HAN import HAN
from data_process import  build_data_cv, make_idx_data_cv

drop_keep_prob = 0.5
nb_epoch = 80
batch_size = 32
max_num = 150
max_len = 50
import pickle

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# X_train, Y_train, X_test, Y_test = make_idx_data_cv(revs, "data/vocab.pickle", 0, 100, 100)

# 0.54530150 y0, max_num: 150 , max_num: 50, lstm
# 0.56561 y0 10-折 learning_rate: 1, max_num: 100 , max_num: 100
# 0.57623 y0 10-折 learning_rate: 5, max_num: 150 , max_num: 50
# 0.5689  y0 10-折 learning_rate: 5, max_num: 150 , max_num: 50, attention

# 0.5657  y0 10-折 learning_rate: 5, Adade, max_num: 150 , max_num: 50, batch:32, aux:y1


# 0.55609 y0 10-折 learning_rate: 1e-4, adam, max_num: 150 , max_num: 50, batch:32
# 0.569109 y0 10-折 learning_rate: 10, max_num: 150 , max_num: 50
# 0.59325 y4 10-折 learning_rate: 10, max_num: 150 , max_num: 50
# 0.60934 y4 10-折 adam learning_rate: 1e-4, max_num: 150 , max_num: 50
# 0.601387 y4 10-折 adam learning_rate: 1e-3, max_num: 150 , max_num: 50

def tranform_y(y):
    num_class = np.max(y)
    Y = []
    for one_y in y:
        labels = [0] * (num_class + 1)
        labels[one_y] = 1
        Y.append(labels)
    Y = np.array(Y)
    return Y

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def train_cv(datasets, data_split, max_num, max_len, grad_clip=5):

    print ('CV:' + str(data_split + 1))

    np.random.seed(3306)
    tf.set_random_seed(3306)

    X_train, Y_train, X_test, Y_test, _, _, = make_idx_data_cv(datasets, "data/vocab.pickle", data_split, max_num, max_len)

    Y_train_0 = tranform_y(Y_train[0])
    Y_test_0 = tranform_y(Y_test[0])

    Y_train_1 = tranform_y(Y_train[1])
    Y_test_1 = tranform_y(Y_test[1])

    Y_train_2 = tranform_y(Y_train[2])
    Y_test_2 = tranform_y(Y_test[2])

    Y_train_3 = tranform_y(Y_train[3])
    Y_test_3 = tranform_y(Y_test[3])

    Y_train_4 = tranform_y(Y_train[4])
    Y_test_4 = tranform_y(Y_test[4])

    test_accuary = 0
    # print Y_train

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = False
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            han = HAN(
                num_classes=2,
                embedding_size=300,
                l2_reg_lambda=0,
                max_num = max_num,
                max_len = max_len
            )

            global_step = tf.Variable(0, name="global_step", trainable=False)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate=5, rho=0.95, epsilon=1e-08)  # epsilon=1e-06

            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            # optimizer = tf.train.AdagradOptimizer(0.1)
            # optimizer = tf.train.AdamOptimizer(1e-3)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(han.loss, tvars), grad_clip)
            grads_and_vars = tuple(zip(grads, tvars))
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch_0, y_batch_1, y_batch_2, y_batch_3, y_batch_4):
                # print "len", len(x_batch)
                feed_dict = {
                    han.input_w_x:x_batch,
                    han.input_y0:y_batch_0,
                    han.input_y1: y_batch_1,
                    han.input_y2: y_batch_2,
                    han.input_y3: y_batch_3,
                    han.input_y4: y_batch_4,
                    # han.input_y: y_batch,
                    han.dropout_keep_prob: drop_keep_prob,
                    han.batch_size: len(x_batch)
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, han.loss, han.accuracy],
                    feed_dict
                )
                return loss, accuracy

            # def dev_step(x_batch, y_batch_0, y_batch_1, y_batch_2, y_batch_3, y_batch_4):
            #     feed_dict = {
            #         han.input_w_x: x_batch,
            #         han.input_y: y_batch,
            #         han.dropout_keep_prob: 1
            #     }
            #     step, loss, accuracy = sess.run(
            #         [global_step, han.loss, han.accuracy],
            #         feed_dict
            #     )
            #     return loss, accuracy

            def dev(X, y0, y1, y2, y3, y4, batch_size):

                batches = batch_iter(list(zip(X, y0, y1, y2, y3, y4)), batch_size=batch_size, num_epochs=1,
                                     shuffle=False)
                all_predictions = []
                for x_batch in batches:
                    batch_X, y_batch_0, y_batch_1, y_batch_2, y_batch_3, y_batch_4 = zip(*x_batch)
                    feed_dict = {
                        han.input_w_x: batch_X,
                        han.input_y0: y_batch_0,
                        han.input_y1: y_batch_1,
                        han.input_y2: y_batch_2,
                        han.input_y3: y_batch_3,
                        han.input_y4: y_batch_4,
                        # han.input_y: batch_y,
                        han.dropout_keep_prob: 1,
                        han.batch_size: len(batch_X)
                    }
                    loss, predictions = sess.run(
                        [han.loss, han.y_pred],
                        feed_dict
                    )
                    all_predictions = np.concatenate([all_predictions, predictions])

                y = np.argmax(y0, axis=1)
                correct_predictions = float(sum(all_predictions == y))
                accuary = correct_predictions / float(len(y))
                return accuary


            for e in range(nb_epoch):
                epoch_starttime = time.time()
                i = 0
                while i < len(X_train):
                    if i + batch_size < len(X_train):
                        batch_xs = X_train[i: i+ batch_size]
                        # batch_ys = Y_train[i: i+ batch_size]
                        batch_ys_0 = Y_train_0[i: i + batch_size]
                        batch_ys_1 = Y_train_1[i: i + batch_size]
                        batch_ys_2 = Y_train_2[i: i + batch_size]
                        batch_ys_3 = Y_train_3[i: i + batch_size]
                        batch_ys_4 = Y_train_4[i: i + batch_size]
                    else:
                        batch_xs = X_train[i:]
                        # batch_ys = Y_train[i:]
                        batch_ys_0 = Y_train_0[i:]
                        batch_ys_1 = Y_train_1[i:]
                        batch_ys_2 = Y_train_2[i:]
                        batch_ys_3 = Y_train_3[i:]
                        batch_ys_4 = Y_train_4[i:]

                    i += batch_size
                    loss, accuary = train_step(batch_xs, batch_ys_0, batch_ys_1, batch_ys_2, batch_ys_3, batch_ys_4)
                    print ("epoch {:g}, loss {:g}, acc {:g}".format((e+1), loss, accuary))
                t_accuary = dev(X_test, Y_test_0, Y_test_1, Y_test_2, Y_test_3, Y_test_4, batch_size)
                if test_accuary < t_accuary:
                    test_accuary = t_accuary
                print ("Epoch traing time: %.1fs" % (time.time() - epoch_starttime))
                print ("test, epoch {:g}, acc {:g}, best acc {:g}".format((e+1), t_accuary, test_accuary))
                print ("Epoch time: %.1fs" % (time.time() - epoch_starttime))
    return test_accuary


def main():
    final = []
    # dataset, _ = build_data_cv('data/essays.csv', cv=10)
    dataset = pickle.load(open("data/dataset.pickle", 'rb'))
    for i in range(10):
        test_accuary = train_cv(dataset, i, max_num, max_len, grad_clip=5)
        final.append(test_accuary)
    print ('Final Test Accuracy:' + str(np.mean(final)))


main()




























