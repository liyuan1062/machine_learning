import tensorflow as tf
from resnet_util import *
from data_utils import *
import numpy as np


NUM_CLASS = 10

TRAINING = tf.Variable(initial_value=True, dtype=tf.bool, trainable=False)


def identity_block(X_input, kernel_size, filter_size, block):
    """
    Implementation of the identity block as defined in Figure 3
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    block -- string/character, used to name the layers, depending on their position in the network
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'conv_' + block
    bn_name_base = 'bn_' + block

    with tf.name_scope("id_block_%s" % block):
        X_shortcut = X_input

        # First component of main path
        x = tf.layers.conv2d(X_input, filter_size, kernel_size, strides=(1, 1),
                             name=conv_name_base+'_a', padding='same')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'_a', training=TRAINING)
        x = tf.nn.relu(x)

        # Second component of main path
        x = tf.layers.conv2d(x, filter_size, kernel_size, strides=(1, 1),
                             name=conv_name_base+'_b', padding='same')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'_b', training=TRAINING)

        # shotcuts
        if X_shortcut.shape[-1] * 2 == x.shape[-1]:
            X_shortcut = tf.layers.conv2d(X_shortcut, x.shape[-1], (1,1),
                                      strides=(1, 1), name=conv_name_base + '_shortcuts')
            X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '_shortcuts', training=TRAINING)
        X_add_shortcut = tf.add(x, X_shortcut)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def ResNet50_reference(X, classes=10):
    """
    Implementation of the popular ResNet50 the following architecture:
    1+2n+2n+2n+1, n=9
    1: 3x3 con2d, filter size: 16, output size: 32x32x16
    2n: 3x3 con2d, filter size: 16, output size: 32x32x16

    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    Arguments:
    Returns:
    """

    # x = tf.pad(X, tf.constant([[0, 0],[3, 3,], [3, 3], [0, 0]]), "CONSTANT")

    # assert(x.shape == (x.shape[0], 38, 38, 3))
    # stage 1
    x = tf.layers.conv2d(X, filters=16, kernel_size=(3, 3), strides=(1, 1), name='conv1', padding='same')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_conv1')
    x = tf.nn.relu(x)


    # first 2n
    for i in range(9):
        x = identity_block(x, kernel_size=(3, 3), filter_size=16, block='first_2n_%s' % i)

    # second 2n
    for i in range(9):
        x = identity_block(x, kernel_size=(3, 3), filter_size=32, block='second_2n_%s' % i)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2))

    # third 2n
    for i in range(9):
        x = identity_block(x, kernel_size=(3, 3), filter_size=64, block='third_2n_%s' % i)

    x = tf.layers.average_pooling2d(x, pool_size=(2, 2), strides=(2, 2))
    fc_output = output_layer(x, NUM_CLASS)
    flatten = tf.layers.flatten(fc_output, name='flatten')
    logits = tf.layers.dense(flatten, units=10, activation=tf.nn.softmax)
    return logits


def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    # regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)

    fc_w = tf.get_variable(name='fc_weights', shape=[input_dim, num_labels],
                           initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = tf.get_variable(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def main():
    path = '../../data/cifar-10-batches-py'
    orig_data = load_CIFAR10(path)
    global TRAINING


    classes = 10
    X_train, Y_train, X_test, Y_test = process_orig_datasets(orig_data, classes)
    # X_train, Y_train, X_test, Y_test = load_CIFAR10(path)


    m, H_size, W_size, C_size = X_train.shape

    X = tf.placeholder(tf.float32, shape=(None, H_size, W_size, C_size), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, classes), name='Y')

    logits = ResNet50_reference(X)

    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=logits))

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(Y, axis=1)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size=128)

        for i in range(40000):
            X_mini_batch, Y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]
            _, cost_sess = sess.run([train_op, loss], feed_dict={X: X_mini_batch, Y: Y_mini_batch})

            if i % 200 == 0:
                print(i, cost_sess)

        sess.run(tf.assign(TRAINING, False))

        training_acur = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train})
        testing_acur = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})
        print("traing acurracy: ", training_acur)
        print("testing acurracy: ", testing_acur)


if __name__ == '__main__':
    main()