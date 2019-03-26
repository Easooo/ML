#author : Easo
import tensorflow as tf
import numpy as np
x = tf.constant([[1],[2],[3],[4]],dtype=tf.float32)
y_true = tf.constant([[0],[-1],[-2],[-3]],dtype=tf.float32)
l_model = tf.layers.Dense(units=1)

y_pred = l_model(x)
loss = tf.losses.mean_squared_error(labels=y_true,predictions=y_pred)
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(2000):
    sess.run(train)
    print(sess.run(loss))

print(sess.run(y_pred))
