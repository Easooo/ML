# -*- coding: utf-8 -*-
#author: easo
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

#获取数据
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#定义层
def add_layer(inputs,in_size,out_size,act_func=None):
    """
    argvs:
        inputs:输入数据
        in_size:层输入的神经元数量
        out_size:输出的神经元数量
        act_func:激活函数，默认为None

    return值:
        outputs:该层的输出
    """
    Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
    biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
    wx_add_b = tf.matmul(inputs,Weights) + biases
    if act_func is None:
        outputs = wx_add_b
    else:
        outputs = act_func(wx_add_b)
    return outputs

#定义准确率
def compute_acc(v_xs,v_ys):
    """
    argvs:
        v_xs:输入数据
        v_ys:输出数据

    return值:
        result:准确率的结果
    """
    global predict
    y_pre = sess.run(predict,feed_dict={xs:v_xs})
    correct_predict = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    acc = tf.reduce_mean(tf.cast(correct_predict,tf.float32))
    result = sess.run(acc,feed_dict={xs:v_xs,ys:v_ys})
    return result


#定义占位符
xs = tf.placeholder(tf.float32,[None,784]) #28*28
ys = tf.placeholder(tf.float32,[None,10]) #10个输出 （0 - 9）

#添加层
predict = add_layer(xs,784,10,act_func=tf.nn.softmax)

#定义loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(predict),reduction_indices=[1])) #交叉熵
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

#开始训练
for i in range(5000):
    batch_xs,batch_ys = mnist.train.next_batch(100) #分批次进行训练
    sess.run(train,feed_dict={xs:batch_xs,ys:batch_ys})
    #计算准确率
    if i % 50 == 0: 
        print(compute_acc(mnist.test.images,mnist.test.labels))
