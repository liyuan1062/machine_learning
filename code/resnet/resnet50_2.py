import tensorflow as tf
from resnet_util import *
from data_utils import *
import numpy as np


NUM_CLASS = 10

TRAINING = tf.Variable(initial_value=True, dtype=tf.bool, trainable=False)

HPS = {'momentum': 0.9, 'batch_size': 28}


def _conv(name, x, filter_size, strides):
    """Convolution."""
    n = 3 * 3 * filter_size
    x = tf.layers.conv2d(x, filter_size, kernel_size=(3, 3), strides=strides,  name=name, padding='same',
                         kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
    return x


def identity_block(x_input, filter_size, resnet_name, strides, bn_relu_shortcuts=False):
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
    conv_name_base = 'conv_' + resnet_name
    bn_name_base = 'bn_' + resnet_name

    with tf.name_scope("identiry_block_%s" % resnet_name):
        if bn_relu_shortcuts:
            x = tf.layers.batch_normalization(x_input, axis=3, momentum=HPS['momentum'], name=bn_name_base + '_a',
                                              training=TRAINING)
            x = tf.nn.relu(x)
            x_shortcut = x
        else:
            x_shortcut = x_input
            x = tf.layers.batch_normalization(x_input, axis=3, momentum=HPS['momentum'], name=bn_name_base + '_a',
                                              training=TRAINING)
            x = tf.nn.relu(x)


        # first conv
        x = _conv(name=conv_name_base+'_a', x=x, filter_size=filter_size, strides=strides)

        # second conv
        x = tf.layers.batch_normalization(x, axis=3, momentum=HPS['momentum'], name=bn_name_base+'_b', training=TRAINING)
        x = tf.nn.relu(x)
        x = _conv(name=conv_name_base+'_b', x=x, filter_size=filter_size, strides=(1, 1))
        # shotcuts
        if x_shortcut.shape[-1] * 2 == x.shape[-1]:
            x_shortcut = tf.layers.average_pooling2d(x_shortcut, (2, 2), (2, 2), 'VALID')
            x_shortcut = tf.pad(x_shortcut, [[0, 0], [0, 0], [0, 0],
                         [(x.shape[-1] - x_shortcut.shape[-1]) // 2, (x.shape[-1] - x_shortcut.shape[-1]) // 2]])
        x_add_shortcut = tf.add(x, x_shortcut)

    return x_add_shortcut


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

    # init conv
    x = _conv(name='conv1', x=X, filter_size=16, strides=(1, 1))

    # first 2n
    x = identity_block(x, filter_size=16, resnet_name='first_2n_0', strides=(1, 1), bn_relu_shortcuts=True)
    for i in range(1, 9):
        x = identity_block(x, filter_size=16, resnet_name='first_2n_%s' % i, strides=(1, 1))

    # second 2n
    x = identity_block(x, filter_size=32, resnet_name='second_2n_0', strides=(2, 2))
    for i in range(1, 9):
        x = identity_block(x, filter_size=32, resnet_name='second_2n_%s' % i, strides=(1, 1))

    # third 2n
    x = identity_block(x, filter_size=64, resnet_name='third_2n_0', strides=(2, 2))
    for i in range(1, 9):
        x = identity_block(x, filter_size=64, resnet_name='third_2n_%s' % i, strides=(1, 1))

    x = tf.layers.batch_normalization(x, axis=3, momentum=HPS['momentum'], name="last_bn", training=TRAINING)
    x = tf.nn.relu(x)
    x = tf.layers.average_pooling2d(x, pool_size=(2, 2), strides=(2, 2))
    # logits = _fully_connected(x, NUM_CLASS)
    flatten = tf.layers.flatten(x, name='flatten')
    logits = tf.layers.dense(flatten, units=10, activation=tf.nn.softmax,
                              kernel_initializer=tf.variance_scaling_initializer(mode='fan_avg'))
    return logits


def _fully_connected(x, out_dim):
    """FullyConnected layer for final output."""
    dw_shape = [int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3]), out_dim]
    x = tf.reshape(x, [HPS['batch_size'], -1])
    w = tf.get_variable(
        'DW', dw_shape, initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)


def main():
    path = '../../data/cifar-10-batches-py'
    # path = '/aiml/data/cifar-10-batches-py'
    orig_data = load_CIFAR10(path)
    global TRAINING


    classes = 10
    # X_train, Y_train, X_test, Y_test = process_orig_datasets(orig_data, classes)
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

        mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size=HPS['batch_size'])

        for i in range(10):
            X_mini_batch, Y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]
            _, cost_sess = sess.run([train_op, loss], feed_dict={X: X_mini_batch, Y: Y_mini_batch})

            if i % 1 == 0:
                print(i, cost_sess)

        sess.run(tf.assign(TRAINING, False))

        print("start training step!")
        # training_acur = sess.run(accuracy, feed_dict={X: X_train[0:40000], Y: Y_train[0:40000]})
        training_acur = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train})
        print("training step done!")
        testing_acur = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})
        print("traing acurracy: ", training_acur)
        print("testing acurracy: ", testing_acur)


if __name__ == '__main__':
    main()