# -*- coding: utf-8 -*-
#author: easo

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#导入数据
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

lr = 0.001 #学习速率
training_iters = 100000
batch_size = 128
display_step = 10

n_inputs = 28 #MNIST数据集的输入（28*28）
n_steps = 28 #28次
n_hidden_unis = 128
n_classes = 10


#定义输入
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

#定义权值
weights = {
    #(28,128)
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_unis])),
    #(128,10)
    'out':tf.Variable(tf.random_normal([n_hidden_unis,n_classes]))
}

#定义偏置
biases = {
    #(128,)
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_unis,])),
    #(10,)
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

#定义RNN
def RNN(X,weights,biases):
    pass
    return None

predict = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict,y))

#定义优化器
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(cost)

#定义正确率
correct_predict = tf.equal(tf.argmax(predict,1),tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correct_predict,tf.float32))

#初始化全局变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={x:batch_xs,y:batch_ys})
        if step % 20 == 0:
            print(sess.run(acc,feed_dict={x:batch_xs,y:batch_ys}))
        step += 1
