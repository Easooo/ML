# -*- coding: utf-8 -*-
"""
author:Easo
"""

import tensorflow as tf
import numpy as np

#定义数据
x_data = np.random.rand(100).astype(np.float32)
#print(x_data)
y_data = x_data*0.1 + 0.3

#定义权值和偏置
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y_predit = x_data*Weights + biases
loss = tf.reduce_mean(tf.square(y_predit-y_data))

#优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#创建会话
sess = tf.Session()
sess.run(init)

for setp in range(201):
    sess.run(train)
    if setp % 20 == 0:
        print(setp,sess.run(Weights),sess.run(biases))
sess.close()