# coding=utf-8

import tensorflow as tf


def w_variable(shape, lambda_r=0):
    # initial weight with normal distribution with stddev=0.1
    initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if lambda_r:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lambda_r)(initial))
    return initial

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    # 卷积不改变输出的shape
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')