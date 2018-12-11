# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

inputA = tf.placeholder(tf.float32)
inputB = tf.placeholder(tf.float32)

output = tf.multiply(inputA,inputB)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={inputA:[8.],inputB:[5.]}))
sess.close()