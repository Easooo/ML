# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

matA = tf.constant([[5.,5.]]) #1*2
matB = tf.constant([[3.],
                    [3.]])   #2*1

result = tf.matmul(matA,matB)

sess = tf.Session()
finish = sess.run(result)
print(finish)
sess.close()

with tf.Session() as sess:
    finish2 = sess.run(result)
    print(finish2)
sess.close()