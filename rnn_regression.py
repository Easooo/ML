# -*- coding: utf-8 -*-
#author: easo

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006
BATCH_START_TEST = 0

def get_batch():
    """
    获得新的batch,生成数据
    return:
        [序列、结果、横轴值]
    """
    global BATCH_START,TIME_STEPS
    #xs -> (50batch,20steps)
    xs = np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE,TIME_STEPS)) #50行 20列
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    plt.plot(xs[0,:],res[0,:],'r',xs[0,:],seq[0,:],'b--')
    plt.show()
    return [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]

class LSTMRNN(object):
    def __init__(self,n_steps,input_size,output_size,cell_size,batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32,[None,n_steps,input_size],name='xs')
            self.ys = tf.placeholder(tf.float32,[None,n_steps,output_size],name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            optimier = tf.train.AdamOptimizer(LR)
            self.train_op = optimier.minimize(self.cost)

    def add_input_layer(self):
        pass
    
    def add_cell(self):
        pass
    
    def add_output_layer(self):
        pass
    
    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred,[-1],name='reshape_pred')],
            [tf.reshape(self.ys,[-1],name='reshape')],
            [tf.ones([self.batch_size*self.n_steps],dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.msr_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses,name='losses_sum'),
                tf.cast(self.batch_size,tf.float32),
                name='average_cost'
            )
            tf.summary.scalar('cost',self.cost)

    def msr_error(self,y_pre,y_target):
        return tf.square(tf.subtract(y_pre,y_target))

    def _weight_variable(self,shape,name='weights'):
        
    
