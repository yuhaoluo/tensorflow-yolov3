import numpy as np
import tensorflow as tf
import common as common


class TestNet(object):
    def __init__(self,input,is_train=True):
        self.output = self.build(input,is_train)

    def build(self,input_data,is_train):
        self.trainable = is_train
        #with tf.name_scope()
        #input_data = common.convolutional(input_data, (1, 1, 3, 5), self.trainable, 'conv1')
        #output = common.convolutional(input_data,(1,1,5,1),self.trainable,'conv2')
        input_data = common.test_conv(input_data, (1, 1, 3, 5), self.trainable, 'conv1')
        output = common.test_conv(input_data,(1,1,5,1),self.trainable,'conv2')
        return output

tf.reset_default_graph()

input_data = tf.placeholder(shape=[None,10,10,3],dtype=tf.float32)
net = TestNet(input_data,True)
moving_ave = tf.train.ExponentialMovingAverage(0.995).apply(tf.trainable_variables())
restore_vars = []
for var in tf.trainable_variables():
    var_name = var.op.name
    if 'FRN' in var_name.split('/') or 'bias' in var_name.split('/'):  #batch_normalization
        continue
    restore_vars.append(var)
    print('add '+ var_name )

loader = tf.train.Saver(restore_vars)
saver = tf.train.Saver(tf.global_variables())
#ema_obj = tf.train.ExponentialMovingAverage(0.995)
#for var in ema_obj.variables_to_restore():
#    print(var)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #sess.run(moving_ave)
    #loader.restore(sess,'./checkpoint/mov_frn.ckpt-1')
    #print('resore')
    for var in tf.trainable_variables():
        var_name = var.op.name
        print(var_name,var.shape)

    ckpt_file = './checkpoint/mov_frn.ckpt'
    saver.save(sess, ckpt_file, global_step=3)
    

# 从checkpoint 文件读取变量
with tf.Session() as sess:
    for var_name,value in tf.contrib.framework.list_variables('./checkpoint/mov_frn.ckpt-3'):
        var = tf.contrib.framework.load_variable('./checkpoint/mov_frn.ckpt-3',var_name)
        print(var_name)
