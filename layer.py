# -*- coding: utf-8 -*-
#author: easo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#定义层
def add_layer(inputs,in_size,out_size,act_func=None):
    """
    inputs:输入数据
    in_size:层输入的神经元数量
    out_size:输出的神经元数量
    act_func:激活函数，默认为None
    """
    with tf.name_scope('Layer'):
        with tf.name_scope('Weights'):
             Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope('wx_add_b'):
            wx_add_b = tf.matmul(inputs,Weights) + biases
        if act_func is None:
            outputs = wx_add_b
        else:
            outputs = act_func(wx_add_b)
        return outputs

x_data = np.linspace(-2,2,300)[:,np.newaxis]
in_noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + in_noise
print(x_data)

#定义占位符，相当于形参
with tf.name_scope('Inputs'):
    xs = tf.placeholder(dtype=tf.float32,shape=[None,1],name='x_input')
    ys = tf.placeholder(dtype=tf.float32,shape=[None,1],name='y_input')

l1 = add_layer(xs,1,10,act_func=tf.nn.relu)
predict = add_layer(l1,10,1,act_func=None)

#计算loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predict),reduction_indices=[1]))
with tf.name_scope('train'):
    tf_train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化所有变量
init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("./logs",sess.graph)
sess.run(init)

#可视化
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(10000):
    sess.run(tf_train,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(p_lines[0])
        except Exception:
            pass
        prediction = sess.run(predict,feed_dict={xs:x_data})
        p_lines = ax.plot(x_data,prediction,'r-')
        plt.pause(0.15)

# with tf.Session() as sess:
#     print(sess.run(tf.reduce_sum(x_data,reduction_indices=[1])))
