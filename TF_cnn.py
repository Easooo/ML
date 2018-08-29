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
    y_pre = sess.run(predict,feed_dict={xs:v_xs,keep_prob:1})
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

#定义池化层
def max_pool_2x2(x):
    """
    argvs:
        x:输入的信息
    return:
        tf的最大池化
    """
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义数据
xs = tf.placeholder(tf.float32,[None,784]) #28*28
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_img = tf.reshape(xs,[-1,28,28,1])

#定义 层1
W_conv1 = weight_var([5,5,1,32]) #patch:5*5，in size:1，out size: 32 (32个卷积核)
b_conv1 = bias_var([32])
h_conv1 = tf.nn.relu(conv2d(x_img,W_conv1) + b_conv1) #使其非线性化 out: 28*28*32
h_pool1 = max_pool_2x2(h_conv1) #out：14*14*64

#定义层2
W_conv2 = weight_var([5,5,32,64]) #patch:5*5，in size:32，out size: 64 (64个卷积核)
b_conv2 = bias_var([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) #使其非线性化 out: 14*14*64
h_pool2 = max_pool_2x2(h_conv2) #out：7*7*64

#定义连接层1
W_fc1 = weight_var([7*7*64,1024])
b_fc1 = bias_var([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) #7,7,64 -> 7*7*64
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#定义连接层2
W_fc2 = weight_var([1024,10]) # 0 ~ 9 十个数字
b_fc2 = bias_var([10])
predict = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

#计算交叉熵
croos_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(predict),reduction_indices=[1]))

#定义优化器
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(croos_entropy)

#存储
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(3000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i % 100 == 0:   
        print("准确率: ",compute_acc(mnist.test.images,mnist.test.labels))

#保存
saver.save(sess,"./cnn_save/save-mnist.ckpt")
sess.close()
