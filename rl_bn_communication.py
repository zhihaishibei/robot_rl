#!/usr/bin/env python

'''
this routine is successfully!
decay = 0.9
used batch normalization
don't use spatial softmax
training result is saved in model0_9
add communication function
this rooutine can run communication with robot successfully
'''

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import socket
import random
import pickle

#from tensorflow.contrib import learn
#from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
# prepare data
img_refer='/home/wzl/design/refer.tif'
#8603328

os.chdir('/home/wzl/design/img_data_mix_seq_train/')
imgnames = []
f = open("imdir.txt")  
line = f.readline()
imgnames.append(line[0:-1])
while line: 
    line = f.readline() 
    imgnames.append(line[0:-1])
f.close()  
imgnames.pop()
tempimg = np.array(Image.open(imgnames[0]))

test_img_name = []
f_test = open("test_data.txt")
test_line = f_test.readline()
test_img_name.append(test_line[0:-1])
while test_line:
    test_line = f_test.readline()
    test_img_name.append(test_line[0:-1])
f_test.close()
test_img_name.pop()


label=np.loadtxt('/home/wzl/design/label_mix_seq_train.txt')
testlabel = np.loadtxt('/home/wzl/design/label_mix_seq_test.txt')

nbatch = 1 
height = tempimg.shape[0]
width = tempimg.shape[1]
channel = 1
ntrain = len(imgnames)
decay = 0.9

tf.device('/gpu:1')
model_path = "/home/wzl/design/model0_9/"

# network setup
s = socket.socket()         # Create a socket object
#host = socket.gethostname() # Get local machine name
host = ''                    # Get local machine name
port = 21567  # Reserve a port for your service.
s.bind((host, port))


#params
conv1_w = tf.Variable(tf.random_normal(([3,3,1,96]),stddev=np.sqrt(2.0/9/1)),dtype="float32")
conv1_b = tf.Variable(tf.zeros([96]),dtype="float32")
conv2_w = tf.Variable(tf.random_normal(([3,3,96,256]),stddev=np.sqrt(2.0/9/96)),dtype="float32")
conv2_b = tf.Variable(tf.zeros([256]),dtype="float32")
conv3_w = tf.Variable(tf.random_normal(([3,3,256,384]),stddev=np.sqrt(2.0/9/256)),dtype="float32")
conv3_b = tf.Variable(tf.zeros([384]),dtype="float32")
conv4_w = tf.Variable(tf.random_normal(([3,3,384,384]),stddev=np.sqrt(2.0/9/384)),dtype="float32")
conv4_b = tf.Variable(tf.zeros([384]),dtype="float32")
conv5_w = tf.Variable(tf.random_normal(([3,3,384,256]),stddev=np.sqrt(2.0/9/384)),dtype="float32")
conv5_b = tf.Variable(tf.zeros([256]),dtype="float32")

conv1_w_ = tf.Variable(tf.random_normal(([3,3,1,96]),stddev=np.sqrt(2.0/9/1)),dtype="float32")
conv1_b_ = tf.Variable(tf.zeros([96]),dtype="float32")
conv2_w_ = tf.Variable(tf.random_normal(([3,3,96,256]),stddev=np.sqrt(2.0/9/96)),dtype="float32")
conv2_b_ = tf.Variable(tf.zeros([256]),dtype="float32")
conv3_w_ = tf.Variable(tf.random_normal(([3,3,256,384]),stddev=np.sqrt(2.0/9/256)),dtype="float32")
conv3_b_ = tf.Variable(tf.zeros([384]),dtype="float32")
conv4_w_ = tf.Variable(tf.random_normal(([3,3,384,384]),stddev=np.sqrt(2.0/9/384)),dtype="float32")
conv4_b_ = tf.Variable(tf.zeros([384]),dtype="float32")
conv5_w_ = tf.Variable(tf.random_normal(([3,3,384,256]),stddev=np.sqrt(2.0/9/384)),dtype="float32")
conv5_b_ = tf.Variable(tf.zeros([256]),dtype="float32")


conv6_w = tf.Variable(tf.random_normal(([3,3,512,256]),stddev=np.sqrt(2.0/9/512)),dtype="float32")
conv6_b = tf.Variable(tf.zeros([256]),dtype="float32")
conv7_w = tf.Variable(tf.random_normal(([3,3,256,128]),stddev=np.sqrt(2.0/9/256)),dtype="float32")
conv7_b = tf.Variable(tf.zeros([128]),dtype="float32")

conv8_w = tf.Variable(tf.random_normal(([3,3,128,128]),stddev=np.sqrt(2.0/9/128)),dtype="float32")
conv8_b = tf.Variable(tf.zeros([128]),dtype="float32")

conv9_w = tf.Variable(tf.random_normal(([3,3,128,128]),stddev=np.sqrt(2.0/9/128)),dtype="float32")
conv9_b = tf.Variable(tf.zeros([128]),dtype="float32")

fc1_w = tf.Variable(tf.random_uniform([3072,64],-1/np.sqrt(3072),1/np.sqrt(3072)))
fc1_b = tf.Variable(tf.random_uniform([64],0,1e-6))

fc2_w = tf.Variable(tf.random_uniform([64,6],-1/np.sqrt(64),1/np.sqrt(64)))
fc2_b = tf.Variable(tf.random_uniform([6],0,1e-4))


#batch_normalization parameter
data_mean = tf.Variable(tf.constant(0.0, shape=[1], dtype="float"), trainable=False)
data_var = tf.Variable(tf.constant(0.0, shape=[1], dtype="float"), trainable=False)

v_zero = tf.Variable(tf.constant(0, shape=[1], dtype="float"), trainable=False)
v_ones = tf.Variable(tf.constant(1, shape=[1], dtype="float"), trainable=False)

mv_mean1 = tf.Variable(tf.constant(0.0, shape=[96], dtype="float"), trainable=False)
mv_var1 = tf.Variable(tf.constant(0.0, shape=[96], dtype="float"), trainable=False)

bn_offset1 = tf.Variable(tf.constant(0,shape=[96],dtype="float"))
bn_scale1  = tf.Variable(tf.constant(1,shape=[96],dtype="float"))

mv_mean2 = tf.Variable(tf.constant(0.0, shape=[256], dtype="float"), trainable=False)
mv_var2 = tf.Variable(tf.constant(0.0, shape=[256], dtype="float"), trainable=False)

bn_offset2 = tf.Variable(tf.constant(0,shape=[256],dtype="float"))
bn_scale2  = tf.Variable(tf.constant(1,shape=[256],dtype="float"))

mv_mean3 = tf.Variable(tf.constant(0.0, shape=[384], dtype="float"), trainable=False)
mv_var3 = tf.Variable(tf.constant(0.0, shape=[384], dtype="float"), trainable=False)

bn_offset3 = tf.Variable(tf.constant(0,shape=[384],dtype="float"))
bn_scale3  = tf.Variable(tf.constant(1,shape=[384],dtype="float"))

mv_mean4 = tf.Variable(tf.constant(0.0, shape=[384], dtype="float"), trainable=False)
mv_var4 = tf.Variable(tf.constant(0.0, shape=[384], dtype="float"), trainable=False)

bn_offset4 = tf.Variable(tf.constant(0,shape=[384],dtype="float"))
bn_scale4  = tf.Variable(tf.constant(1,shape=[384],dtype="float"))

mv_mean5 = tf.Variable(tf.constant(0.0, shape=[256], dtype="float"), trainable=False)
mv_var5 = tf.Variable(tf.constant(0.0, shape=[256], dtype="float"), trainable=False)

bn_offset5 = tf.Variable(tf.constant(0,shape=[256],dtype="float"))
bn_scale5  = tf.Variable(tf.constant(1,shape=[256],dtype="float"))



data_mean_ = tf.Variable(tf.constant(0.0, shape=[1], dtype="float"), trainable=False)
data_var_ = tf.Variable(tf.constant(0.0, shape=[1], dtype="float"), trainable=False)

v_zero_ = tf.Variable(tf.constant(0, shape=[1], dtype="float"), trainable=False)
v_ones_ = tf.Variable(tf.constant(1, shape=[1], dtype="float"), trainable=False)

mv_mean1_ = tf.Variable(tf.constant(0.0, shape=[96], dtype="float"), trainable=False)
mv_var1_ = tf.Variable(tf.constant(0.0, shape=[96], dtype="float"), trainable=False)

bn_offset1_ = tf.Variable(tf.constant(0,shape=[96],dtype="float"))
bn_scale1_  = tf.Variable(tf.constant(1,shape=[96],dtype="float"))

mv_mean2_ = tf.Variable(tf.constant(0.0, shape=[256], dtype="float"), trainable=False)
mv_var2_ = tf.Variable(tf.constant(0.0, shape=[256], dtype="float"), trainable=False)

bn_offset2_ = tf.Variable(tf.constant(0,shape=[256],dtype="float"))
bn_scale2_  = tf.Variable(tf.constant(1,shape=[256],dtype="float"))

mv_mean3_ = tf.Variable(tf.constant(0.0, shape=[384], dtype="float"), trainable=False)
mv_var3_ = tf.Variable(tf.constant(0.0, shape=[384], dtype="float"), trainable=False)

bn_offset3_ = tf.Variable(tf.constant(0,shape=[384],dtype="float"))
bn_scale3_  = tf.Variable(tf.constant(1,shape=[384],dtype="float"))

mv_mean4_ = tf.Variable(tf.constant(0.0, shape=[384], dtype="float"), trainable=False)
mv_var4_ = tf.Variable(tf.constant(0.0, shape=[384], dtype="float"), trainable=False)

bn_offset4_ = tf.Variable(tf.constant(0,shape=[384],dtype="float"))
bn_scale4_  = tf.Variable(tf.constant(1,shape=[384],dtype="float"))

mv_mean5_ = tf.Variable(tf.constant(0.0, shape=[256], dtype="float"), trainable=False)
mv_var5_ = tf.Variable(tf.constant(0.0, shape=[256], dtype="float"), trainable=False)

bn_offset5_ = tf.Variable(tf.constant(0,shape=[256],dtype="float"))
bn_scale5_  = tf.Variable(tf.constant(1,shape=[256],dtype="float"))



mv_mean6 = tf.Variable(tf.constant(0.0, shape=[256], dtype="float"), trainable=False)
mv_var6 = tf.Variable(tf.constant(0.0, shape=[256], dtype="float"), trainable=False)

bn_offset6 = tf.Variable(tf.constant(0,shape=[256],dtype="float"))
bn_scale6  = tf.Variable(tf.constant(1,shape=[256],dtype="float"))

mv_mean7 = tf.Variable(tf.constant(0.0, shape=[128], dtype="float"), trainable=False)
mv_var7 = tf.Variable(tf.constant(0.0, shape=[128], dtype="float"), trainable=False)

bn_offset7 = tf.Variable(tf.constant(0,shape=[128],dtype="float"))
bn_scale7  = tf.Variable(tf.constant(1,shape=[128],dtype="float"))
#define net model 
def regress_model(x, x_, train=False, d=decay):
    #input_layer_new=tf.reshape(x,[-1,400,300,1])
        
    if train:
        dm, dv = tf.nn.moments(x,axes=[0,1,2])
        tm0 = tf.assign(data_mean, data_mean * d + dm * (1 - d))
        tv0 = tf.assign(data_var, data_var * d + dv * (1 - d))
        with tf.control_dependencies([tm0, tv0]):
            x = tf.nn.batch_normalization(x, dm, dv, v_zero, v_ones, 1e-9)
    else:
        x = tf.nn.batch_normalization(x, data_mean, data_var, v_zero, v_ones, 1e-9)
	
    conv1 = tf.nn.conv2d(x,conv1_w,strides=[1,1,1,1],padding="VALID") 
    if train:
    	bm, bv = tf.nn.moments(conv1, axes=[0, 1, 2])
        tm1 = tf.assign(mv_mean1,mv_mean1 * d + bm * (1 - d))
        tv1 = tf.assign(mv_var1, mv_var1 * d + bv * (1 - d))
        with tf.control_dependencies([tm1,tv1]):
            conv1 = tf.nn.batch_normalization(conv1, bm, bv, bn_offset1, bn_scale1, 1e-9)
    else:
        conv1 = tf.nn.batch_normalization(conv1, mv_mean1, mv_var1, bn_offset1, bn_scale1, 1e-9)
	
      
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")   
    
    conv2 = tf.nn.conv2d(pool1,conv2_w,strides=[1,1,1,1],padding="VALID")
    if train:
        bm, bv = tf.nn.moments(conv2, axes=[0, 1, 2])
        tm2 = tf.assign(mv_mean2,mv_mean2 * d + bm * (1 - d))
        tv2 = tf.assign(mv_var2, mv_var2 * d + bv * (1 - d))
        with tf.control_dependencies([tm2,tv2]):
                conv2 = tf.nn.batch_normalization(conv2, bm, bv, bn_offset2, bn_scale2, 1e-9)
    else:
        conv2 = tf.nn.batch_normalization(conv2, mv_mean2, mv_var2, bn_offset2, bn_scale2, 1e-9)
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    
    conv3 = tf.nn.conv2d(pool2,conv3_w,strides=[1,1,1,1],padding="VALID")  
    if train:
        bm, bv = tf.nn.moments(conv3, axes=[0, 1, 2])
        tm3 = tf.assign(mv_mean3,mv_mean3 * d + bm * (1 - d))
        tv3 = tf.assign(mv_var3, mv_var3 * d + bv * (1 - d))
        with tf.control_dependencies([tm3,tv3]):
            conv3 = tf.nn.batch_normalization(conv3, bm, bv, bn_offset3, bn_scale3, 1e-9)
    else:
        conv3 = tf.nn.batch_normalization(conv3, mv_mean3, mv_var3, bn_offset3, bn_scale3, 1e-9)
    conv3 = tf.nn.relu(conv3)    
    
    conv4 = tf.nn.conv2d(conv3,conv4_w,strides=[1,1,1,1],padding="VALID")
    if train:
        bm, bv = tf.nn.moments(conv3, axes=[0, 1, 2])
        tm4 = tf.assign(mv_mean4,mv_mean4 * d + bm * (1 - d))
        tv4 = tf.assign(mv_var4, mv_var4 * d + bv * (1 - d))
        with tf.control_dependencies([tm4,tv4]):
            conv4 = tf.nn.batch_normalization(conv4, bm, bv, bn_offset4, bn_scale4, 1e-9)
    else:
        conv4 = tf.nn.batch_normalization(conv4, mv_mean4, mv_var4, bn_offset4, bn_scale4, 1e-9)    
    conv4 = tf.nn.relu(conv4)    
    
    conv5 = tf.nn.conv2d(conv4,conv5_w,strides=[1,1,1,1],padding="VALID")
    if train:
        bm, bv = tf.nn.moments(conv5, axes=[0, 1, 2])
        tm5 = tf.assign(mv_mean5,mv_mean5 * d + bm * (1 - d))
        tv5 = tf.assign(mv_var5, mv_var5 * d + bv * (1 - d))
        with tf.control_dependencies([tm5,tv5]):
            conv5 = tf.nn.batch_normalization(conv5, bm, bv, bn_offset5, bn_scale5, 1e-9)
    else:
        conv5 = tf.nn.batch_normalization(conv5, mv_mean5, mv_var5, bn_offset5, bn_scale5, 1e-9)
    conv5 = tf.nn.relu(conv5)
    pool3 = tf.nn.max_pool(conv5,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    
    
    
    #input_layer_origin=tf.reshape(x_,[-1,400,300,1])
    if train:
        dm, dv = tf.nn.moments(x_,axes=[0,1,2])
        tm0_ = tf.assign(data_mean_, data_mean_ * d + dm * (1 - d))
        tv0_ = tf.assign(data_var_, data_var_ * d + dv * (1 - d))
        with tf.control_dependencies([tm0_, tv0_]):
            x_ = tf.nn.batch_normalization(x_, dm, dv, v_zero_, v_ones_, 1e-9)
    else:
        x_ = tf.nn.batch_normalization(x_, data_mean_, data_var_, v_zero_, v_ones_, 1e-9)
    conv1_ = tf.nn.conv2d(x_, conv1_w_,strides=[1,1,1,1],padding="VALID")  
    if train:
        bm, bv = tf.nn.moments(conv1_, axes=[0, 1, 2])
        tm1_ = tf.assign(mv_mean1_,mv_mean1_ * d + bm * (1 - d))
        tv1_ = tf.assign(mv_var1_, mv_var1_ * d + bv * (1 - d))
        with tf.control_dependencies([tm1_,tv1_]):
            conv1_ = tf.nn.batch_normalization(conv1_, bm, bv, bn_offset1_, bn_scale1_, 1e-9)
    else:
        conv1_ = tf.nn.batch_normalization(conv1_, mv_mean1, mv_var1, bn_offset1_, bn_scale1_, 1e-9)
    conv1_ = tf.nn.relu(conv1_)
    pool1_ = tf.nn.max_pool(conv1_,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")   
    
    conv2_ = tf.nn.conv2d(pool1_,conv2_w_,strides=[1,1,1,1],padding="VALID")  
    if train:
        bm, bv = tf.nn.moments(conv2_, axes=[0, 1, 2])
        tm2_ = tf.assign(mv_mean2_,mv_mean2_ * d + bm * (1 - d))
        tv2_ = tf.assign(mv_var2_, mv_var2_ * d + bv * (1 - d))
        with tf.control_dependencies([tm2_,tv2_]):
            conv2_ = tf.nn.batch_normalization(conv2_, bm, bv, bn_offset2_, bn_scale2_, 1e-9)
    else:
        conv2_ = tf.nn.batch_normalization(conv2_, mv_mean2_, mv_var2_, bn_offset2_, bn_scale2_, 1e-9)
    conv2_ = tf.nn.relu(conv2_)
    pool2_ = tf.nn.max_pool(conv2_,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    
    conv3_ = tf.nn.conv2d(pool2_,conv3_w_,strides=[1,1,1,1],padding="VALID") 
    if train:
        bm, bv = tf.nn.moments(conv3_, axes=[0, 1, 2])
        tm3_ = tf.assign(mv_mean3_,mv_mean3_ * d + bm * (1 - d))
        tv3_ = tf.assign(mv_var3_, mv_var3_ * d + bv * (1 - d))
        with tf.control_dependencies([tm3_,tv3_]):
            conv3_ = tf.nn.batch_normalization(conv3_, bm, bv, bn_offset3_, bn_scale3_, 1e-9)
    else:
        conv3_ = tf.nn.batch_normalization(conv3_, mv_mean3_, mv_var3_, bn_offset3_, bn_scale3_, 1e-9)
    conv3_ = tf.nn.relu(conv3_)    
    
    conv4_ = tf.nn.conv2d(conv3_,conv4_w_,strides=[1,1,1,1],padding="VALID")   
    if train:
        bm, bv = tf.nn.moments(conv4_, axes=[0, 1, 2])
        tm4_ = tf.assign(mv_mean4_,mv_mean4_ * d + bm * (1 - d))
        tv4_ = tf.assign(mv_var4_, mv_var4_ * d + bv * (1 - d))
        with tf.control_dependencies([tm4_,tv4_]):
            conv4_ = tf.nn.batch_normalization(conv4_, bm, bv, bn_offset4_, bn_scale4_, 1e-9)
    else:
        conv4_ = tf.nn.batch_normalization(conv4_, mv_mean4_, mv_var4_, bn_offset4_, bn_scale4_, 1e-9)
    conv4_ = tf.nn.relu(conv4_)    
    
    conv5_ = tf.nn.conv2d(conv4_,conv5_w_,strides=[1,1,1,1],padding="VALID")
    if train:
        bm, bv = tf.nn.moments(conv5_, axes=[0, 1, 2])
        tm5_ = tf.assign(mv_mean5_,mv_mean5_ * d + bm * (1 - d))
        tv5_ = tf.assign(mv_var5_, mv_var5_ * d + bv * (1 - d))
        with tf.control_dependencies([tm5_,tv5_]):
            conv5_ = tf.nn.batch_normalization(conv5_, bm, bv, bn_offset5_, bn_scale5_, 1e-9)
    else:
        conv5_ = tf.nn.batch_normalization(conv5_, mv_mean5_, mv_var5_, bn_offset5_, bn_scale5_, 1e-9)
    conv5_ = tf.nn.relu(conv5_)
    pool3_ = tf.nn.max_pool(conv5_,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    
    concat=tf.concat(3,[pool3,pool3_])
        
    
    
    conv6 = tf.nn.conv2d(concat,conv6_w,strides=[1,1,1,1],padding="VALID")  
    if train:
        bm, bv = tf.nn.moments(conv6, axes=[0, 1, 2])
        tm6 = tf.assign(mv_mean6,mv_mean6 * d + bm * (1 - d))
        tv6 = tf.assign(mv_var6, mv_var6 * d + bv * (1 - d))
        with tf.control_dependencies([tm6,tv6]):
            conv6 = tf.nn.batch_normalization(conv6, bm, bv, bn_offset6, bn_scale6, 1e-9)
    else:
        conv6 = tf.nn.batch_normalization(conv6, mv_mean6, mv_var6, bn_offset6, bn_scale6, 1e-9)
    conv6 = tf.nn.relu(conv6)   
    
    conv7 = tf.nn.conv2d(conv6,conv7_w,strides=[1,1,1,1],padding="VALID") 
    if train:
        bm, bv = tf.nn.moments(conv7, axes=[0, 1, 2])
        tm7 = tf.assign(mv_mean7,mv_mean7 * d + bm * (1 - d))
        tv7 = tf.assign(mv_var7, mv_var7 * d + bv * (1 - d))
        with tf.control_dependencies([tm7,tv7]):
            conv7 = tf.nn.batch_normalization(conv7, bm, bv, bn_offset7, bn_scale7, 1e-9)
    else:
        conv7 = tf.nn.batch_normalization(conv7, mv_mean7, mv_var7, bn_offset7, bn_scale7, 1e-9)
    conv7 = tf.nn.relu(conv7) 
#    print ('-------------------------------------------')
#    print (conv7.get_shape());
    pool7 = tf.nn.avg_pool(conv7, ksize=[1,3,3,1],strides=[1,3,3,1],padding="SAME")
    #spatial_soft=spatial_softmax(conv7,nbatch,128,42,29)
    conv8=tf.nn.conv2d(pool7,conv8_w,strides=[1,1,1,1],padding = "SAME")
    conv8 = tf.nn.bias_add(conv8,conv8_b)
    conv9=tf.nn.conv2d(conv8,conv9_w,strides=[1,1,1,1],padding = "VALID")
    conv9 = tf.nn.bias_add(conv9,conv9_b)    
    
    pool9 = tf.nn.avg_pool(conv9, ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    
    pool9_flat = tf.reshape(pool9,[nbatch,3072])
    if train:
	pool9_flat=tf.nn.dropout(pool9_flat,0.6)
    fc1 = tf.matmul(pool9_flat,fc1_w)+fc1_b
    fc1 = tf.nn.relu(fc1)
    
    # if train:
        # fc1 = tf.nn.dropout(fc1,0.5)
    
    fc2 = tf.matmul(fc1,fc2_w)+fc2_b
    
    return fc2


x = tf.placeholder(tf.float32, shape=(nbatch, width, height, channel))
x_ = tf.placeholder(tf.float32, shape=(nbatch, width, height, channel))
y = tf.placeholder(tf.float32, shape=(nbatch, 6))
pred = regress_model(x,x_,True)
loss = tf.reduce_sum(tf.pow(y-pred,2))/nbatch
pred_test = regress_model(x,x_,False)
test_loss = tf.reduce_sum(tf.pow(y-pred_test,2))/nbatch

batch = tf.Variable(0, dtype="float", trainable=False)
learning_rate = tf.train.exponential_decay(
  0.001,                # Base learning rate.
  batch * nbatch,  # Current index into the dataset.
  12000,          # Decay step.
  0.9,                # Decay rate.
  staircase=True)

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=batch)

data_ = np.zeros([nbatch,height,width,channel])
img_origin = np.array(Image.open(img_refer))
for i in range(nbatch):
    data_[i,]=img_origin[:,:,np.newaxis]
    i=i+1
data_ = data_.transpose([0,2,1,3])

saver=tf.train.Saver()
training_epochs = 500

referimg = np.array(Image.open('../refer.tif'))
referimg = referimg[np.newaxis,:,:,np.newaxis]
test_refer = referimg.transpose([0,2,1,3])
    
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess,model_path+'061.ckpt')
    
    s.listen(5)
    
    while True:
        c, addr = s.accept()     # Establish connection with client.
        print ('Got connection from'), addr
        print ("Receiving...")
        l = c.recv(1024)
        f = open('../temp.tif','wb')
        while (l):
            f.write(l)
            l = c.recv(1024)
     	f.close()
        print ("Done Receiving")
        tempimg=np.array(Image.open('../temp.tif'))
        tempimg = tempimg[np.newaxis,:,:,np.newaxis]
        test_data = tempimg.transpose([0,2,1,3])
        test_pred = sess.run([pred_test],feed_dict={x:test_data,x_:test_refer})
        test_pred = test_pred[0]
        print ([test_pred[0,0],test_pred[0,1],test_pred[0,2],test_pred[0,3],test_pred[0,4],test_pred[0,5]])
        str_buf = ''
        str_buf = str_buf+str(test_pred[0,0])+" "
        str_buf = str_buf+str(test_pred[0,1])+" "
        str_buf = str_buf+str(test_pred[0,2])+" "
        str_buf = str_buf+str(test_pred[0,3])+" "
        str_buf = str_buf+str(test_pred[0,4])+" "
        str_buf = str_buf+str(test_pred[0,5])+" "
        c.send(str_buf)
        print("send finished!")
        c.close()                # Close the connection
    

