# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

varA = tf.Variable(0,name='counter')
#print(varA.name)
one = tf.constant(1)
new_var = tf.add(varA,one)
update_var = tf.assign(varA,new_var)

#初始化并并激活变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(2):
        sess.run(update_var)
        print(sess.run(new_var),sess.run(varA))
sess.close()
        

