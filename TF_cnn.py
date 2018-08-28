# -*- coding: utf-8 -*-
#author: easo
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#导入数据
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
#定义准确率计算函数

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


#定义权值参数
def weight_var(shape):
    """
    argvs:
        shape:形状，即张量的维度
    return:
        tf格式的变量
    """
    initial = tf.truncated_normal(shape,stddev=0.1) #truncated_normal函数产生正态分布,stddev是标准差
    return tf.Variable(initial)

#定义偏置参数
def bias_var(shape):
    """
    argvs:
        shape:形状，即张量的维度
    return:
        tf格式的变量
    """
    initial = tf.constant(0.1,shape=shape) 
    return tf.Variable(initial)

#定义卷积层
def conv2d(x,W):
    """
    argvs:
        x:输入的图片信息
        W:权值
    return:
        tf的二维卷积神经网络
    """
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME') #strides为步长，[1,1,1,1]代表跨一步

def max_pool_2x2(x):
    pass

#计算交叉熵
croos_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(predict),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(croos_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50 == 0:
        print("准确率: ",compute_acc(mnist.test.images,mnist.test.labels))