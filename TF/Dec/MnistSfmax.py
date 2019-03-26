# -*- coding: utf-8 -*-
#author: easo

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_DATA',one_hot=True)
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.zeros([784,10]))
bias = tf.Variable(tf.zeros([10]))
y_pre = tf.nn.softmax(tf.matmul(xs,W)+bias)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(y_pre),reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()

sess.run(tf.global_variables_initializer())
for i in range(1000):
    x_batch,y_batch = mnist_data.train.next_batch(100)
    sess.run(train,feed_dict={xs:x_batch,ys:y_batch})
    if i % 50 == 0:
        correct_pre = tf.equal(tf.argmax(ys,1),tf.argmax(y_pre,1))
        acc = tf.reduce_mean(tf.cast(correct_pre,tf.float32))
        print(sess.run(acc,feed_dict={xs:mnist_data.test.images,ys:mnist_data.test.labels}))
