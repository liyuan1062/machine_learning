# coding=utf-8

import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import CSVData
import util_nn

file_path = "../../data/MNIST/train.csv"
test_file_path = "../../data/MNIST/test.csv"
print(os.path.abspath(os.curdir))
# train_data = CSVData.CSVData(file_path=file_path)
train_data = CSVData.CSVData(file_path=test_file_path, test=True)
test_image = train_data.normalized_data
test_image_kreas = np.reshape(test_image, [-1,28,28,1])
# test_data = CSVData.CSVData(file_path=test_file_path)
print("training data shape: (%s, %s)" % train_data.shape())
image_size = train_data.shape()[1]
print("length of image pixel: %s" % train_data.shape()[1])

image_width = image_height = np.sqrt(train_data.shape()[1])
print("image size: (%s, %s)" % (image_width, image_height))

label_count = train_data.get_label_count()
# x = tf.placeholder(np.float32, shape=[None, image_size])
x = tf.placeholder(np.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(np.float32, shape=[None, label_count])
# print("label number is %s" % label_count)
# labels_one_hot = train_data.get_one_hot_labels()
# print("label one hot shape is (%s, %s)" % labels_one_hot.shape)
#
# validation_size = 2000
# validation_images = train_data.normalized_data[:validation_size]
# validation_lables = labels_one_hot[:validation_size]
#
# train_images = train_data.normalized_data[validation_size:]
# train_lables = labels_one_hot[validation_size:]

batch_size = 200
# n_batch = len(train_images) // batch_size

# -------- softmax classify--------------------
# 初始化 tensor
# w = tf.Variable(tf.zeros([image_size, label_count]))
# biases = tf.Variable(tf.zeros([label_count], dtype=np.float32))
# h = tf.matmul(x, w) + biases
# prediction = tf.nn.softmax(h)
#
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# #准确度
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# #开始训练
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(50):
#         for batch in range(n_batch):
#             batch_x = train_images[batch*batch_size:(batch+1)*batch_size]
#             batch_y = train_lables[batch*batch_size:(batch+1)*batch_size]
#             sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
#         accuracy_n = sess.run(accuracy, feed_dict={x: validation_images, y: validation_lables})
#         print ("第 {} 轮训练，准确度为 {}".format(epoch+1, accuracy_n))
# -------------------End softmax-----------------------------

# ----------------CNN ----------------------------------
x_image = tf.reshape(x, [-1, 28, 28, 1])
w_conv1 = util_nn.w_variable([5, 5, 1, 32])
b_conv1 = util_nn.bias_variable([32])

# 28x28的图片卷积是步长为1，填充后卷积结果shape不变，
h_conv1 = tf.nn.relu(util_nn.conv2d(x_image, w_conv1) + b_conv1)
# 用2x2 shape来最大值池化, 第一次池化后大小为14x14, 第二次池化后7x7
h_pool1 = util_nn.max_pool_2x2(h_conv1)
w_conv2 = util_nn.w_variable([5, 5, 32, 64])

b_conv2 = util_nn.bias_variable([64])

h_conv2 = tf.nn.relu(util_nn.conv2d(h_pool1, w_conv2) + b_conv2)

h_pool2 = util_nn.max_pool_2x2(h_conv2)
# h_pool2_shape_list = h_pool2.get_shape().as_list()
# nodes_num = h_pool2_shape_list[1] * h_pool2_shape_list[2] * h_pool2_shape_list[3]
# h_pool2_flat = tf.reshape(h_pool2, [h_pool2_shape_list[0], nodes_num])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])


# 构造全连接的神经网络, 1024个神经元
w_fc1 = util_nn.w_variable([7 * 7 * 64, 1024])
b_fc1 = util_nn.bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# 把1024个神经元的输入变为一个10维的输出
w_fc2 = util_nn.w_variable([1024, 10])
b_fc2 = util_nn.bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 保存模型的文件名
global_step = tf.Variable(0, name='global_step', trainable=False)
saver = tf.train.Saver()

init = tf.global_variables_initializer()
# datagen = train_data.data_generator()
# x_keras = np.reshape(train_images, [-1,28,28,1])
# data_keras = datagen.flow(x_keras, train_lables, batch_size=batch_size)
# with tf.Session() as sess:
#     sess.run(init)
#     # 载入之前训练好的模型，又需要采用
#     # saver.restore(sess, 'model.ckpt-12')
#
#     validation_images_keras = np.reshape(validation_images, [-1,28,28,1])
#     for epoch in range(30):
#         # for batch in range(n_batch):
#         #     batch_x = train_images[batch * batch_size: (batch + 1) * batch_size]
#         #     batch_y = train_lables[batch * batch_size: (batch + 1) * batch_size]
#         #     sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
#         i = 0
#         for batch_x, batch_y in data_keras:
#             i += 1
#             sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5})
#             if i == 400:
#                 print("total batch = {}, current batch = {}".format(n_batch, i))
#                 break
#
#         # accuracy_n = sess.run(accuracy, feed_dict={x: validation_images, y: validation_lables, keep_prob: 1.0})
#         accuracy_n = sess.run(accuracy, feed_dict={x: validation_images_keras, y: validation_lables, keep_prob: 1.0})
#         print("第 {} 轮，准确度为 {}".format(epoch + 1, accuracy_n))
#
#         global_step.assign(epoch).eval()
#         saver.save(sess, '../../data/MNIST/model-ckpt/cnn', global_step=global_step)


# testing set
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, '../../data/MNIST/model-ckpt/cnn-25')
    conv_y_predict = y_conv.eval(feed_dict={x:test_image_kreas, keep_prob:1.0})
    test_pred = np.argmax(conv_y_predict, axis=1)
    out_header = 'ImageId,Label'
    out_data = np.int32(np.matrix(np.append(np.arange(1,28001), test_pred)).reshape(2,-1))

    np.savetxt('../../data/MNIST/test_output_25.csv', out_data.T, delimiter=',', header=out_header, comments='')


