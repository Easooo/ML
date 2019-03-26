# -*- coding: utf-8 -*-
#author: easo

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#定义层
def add_layer(inputs,in_size,out_size,n_layer,act_func=None):
	"""
	argvs:
		inputs:输入数据
		in_size:层输入的神经元数量
		out_size:输出的神经元数量
		n_layer: 层的名字，用于tensorboard显示
		act_func:激活函数，默认为None

	return值:
		outputs:该层的输出
	"""
	Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
	biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
	wx_add_b = tf.matmul(inputs,Weights) + biases
	wx_add_b = tf.nn.dropout(wx_add_b,keep_prob)
	if act_func is None:
		outputs = wx_add_b
	else:
		outputs = act_func(wx_add_b)
	tf.summary.histogram(n_layer+'/outputs',outputs)
	return outputs

#定义占位符
keep_prob = tf.placeholder(tf.float32) #dropout变量,数值大小n表示有多少比例的结果需要保持,防止过拟合
xs = tf.placeholder(tf.float32,[None,64]) #8*8
ys = tf.placeholder(tf.float32,[None,10])

#载入数据
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)

#添加层
l1 = add_layer(xs,64,80,'l1',act_func=tf.nn.tanh)
predict = add_layer(l1,80,10,'l2',act_func=tf.nn.softmax)


#定义loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(predict),reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)

#定义优化器
optimizer = tf.train.GradientDescentOptimizer(0.6)

#开始训练
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter("./tmp/train",sess.graph)
test_writer = tf.summary.FileWriter("./tmp/test",sess.graph)

sess.run(tf.global_variables_initializer())

#开始定义训练步骤
for i in range(500):
	sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.5})
	if i % 50 == 0:
		#加入summary中以便可视化
		train_result = sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
		test_result = sess.run(merged,feed_dict={xs:X_test,ys:y_test,keep_prob:1})  #记录result的时候 设置为1
		train_writer.add_summary(train_result,i)
		test_writer.add_summary(test_result,i)