#!/usr/bin/env python

'''
this routine has a class for training
used batch normalization function
learning rate is changable
saver only save the variable needed, it is a dict
'''


import numpy as np
import tensorflow as tf
import os
from PIL import Image
import random

LEARNING_RATE = 0.0001
tau = 0.001
  
class batch_norm:
    def __init__(self,inputs,is_training,sess,parForTarget=None,decay = 0.95,TAU = 0.001,bn_param=None):
        
        self.sess = sess
        self.scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        self.beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        self.pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        self.pop_var = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))        
        self.batch_mean, self.batch_var = tf.nn.moments(inputs,[0,1,2])        
        self.train_mean = tf.assign(self.pop_mean,self.pop_mean * decay + self.batch_mean * (1 - decay))  
        self.train_var = tf.assign(self.pop_var,self.pop_var * decay + self.batch_var * (1 - decay))
                
        def training(): 
            return tf.nn.batch_normalization(inputs, self.batch_mean, self.batch_var, self.beta, self.scale, 0.0000001 )
    
        def testing(): 
            return tf.nn.batch_normalization(inputs, self.pop_mean, self.pop_var, self.beta, self.scale, 0.0000001)
        
        if parForTarget!=None:
            self.parForTarget = parForTarget
            #update scale and beta
            self.updateScale = self.scale.assign(self.scale*(1-TAU)+self.parForTarget.scale*TAU)
            self.updateBeta = self.beta.assign(self.beta*(1-TAU)+self.parForTarget.beta*TAU)
            self.updateTarget = tf.group(self.updateScale, self.updateBeta)
            
        self.bnorm = tf.cond(is_training,training,testing)    
   
class ActorNet:
    def __init__(self,model):
        tf.reset_default_graph()
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            #actor network model parameters:
            self.x = tf.placeholder(tf.float32, shape = [None,400,300,1])
            self.x_= tf.placeholder(tf.float32, shape = [None,400,300,1])
            self.is_training = tf.placeholder(tf.bool, [])
            
            self.conv1_w = tf.Variable(tf.random_normal(([3,3,1,96]),stddev=np.sqrt(2.0/9/1)),dtype="float32",name='conv1_w')
            self.conv2_w = tf.Variable(tf.random_normal(([3,3,96,256]),stddev=np.sqrt(2.0/9/96)),dtype="float32",name='conv2_w')
            self.conv3_w = tf.Variable(tf.random_normal(([3,3,256,384]),stddev=np.sqrt(2.0/9/256)),dtype="float32",name='conv3_w')
            self.conv4_w = tf.Variable(tf.random_normal(([3,3,384,384]),stddev=np.sqrt(2.0/9/384)),dtype="float32",name='conv4_w')
            self.conv5_w = tf.Variable(tf.random_normal(([3,3,384,256]),stddev=np.sqrt(2.0/9/384)),dtype="float32",name='conv5_w')
        
            self.conv1_w_ = tf.Variable(tf.random_normal(([3,3,1,96]),stddev=np.sqrt(2.0/9/1)),dtype="float32",name='conv1_w_')
            self.conv2_w_ = tf.Variable(tf.random_normal(([3,3,96,256]),stddev=np.sqrt(2.0/9/96)),dtype="float32",name='conv2_w_')
            self.conv3_w_ = tf.Variable(tf.random_normal(([3,3,256,384]),stddev=np.sqrt(2.0/9/256)),dtype="float32",name='conv3_w_')
            self.conv4_w_ = tf.Variable(tf.random_normal(([3,3,384,384]),stddev=np.sqrt(2.0/9/384)),dtype="float32",name='conv4_w_')
            self.conv5_w_ = tf.Variable(tf.random_normal(([3,3,384,256]),stddev=np.sqrt(2.0/9/384)),dtype="float32",name='conv5_w_')
        
        
            self.conv6_w = tf.Variable(tf.random_normal(([3,3,512,256]),stddev=np.sqrt(2.0/9/512)),dtype="float32",name='conv6_w')
            self.conv7_w = tf.Variable(tf.random_normal(([3,3,256,128]),stddev=np.sqrt(2.0/9/256)),dtype="float32",name='conv7_w')        
            self.conv8_w = tf.Variable(tf.random_normal(([3,3,128,128]),stddev=np.sqrt(2.0/9/128)),dtype="float32",name='conv8_w')        
            self.conv9_w = tf.Variable(tf.random_normal(([3,3,128,128]),stddev=np.sqrt(2.0/9/128)),dtype="float32",name='conv9_w')
        
            self.fc1_w = tf.Variable(tf.random_uniform([3072,64],-1/np.sqrt(3072),1/np.sqrt(3072)),name='fc1_w')
            self.fc1_b = tf.Variable(tf.random_uniform([64],0,1e-6),name='fc1_b')
        
            self.fc2_w = tf.Variable(tf.random_uniform([64,6],-1/np.sqrt(64),1/np.sqrt(64)),name='fc2_w')
            self.fc2_b = tf.Variable(tf.random_uniform([6],0,1e-4),name='fc2_b')
            #--------------------------------------current input-------------------------------------------------------       
        
            self.conv1 = tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding="VALID",name='conv1') 
            self.conv1_bn = batch_norm(self.conv1,self.is_training,self.sess)
            self.conv1_relu = tf.nn.relu(self.conv1_bn.bnorm)
            self.pool1 = tf.nn.max_pool(self.conv1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")   
        
            self.conv2 = tf.nn.conv2d(self.pool1,self.conv2_w,strides=[1,1,1,1],padding="VALID",name='conv2')
            self.conv2_bn = batch_norm(self.conv2,self.is_training,self.sess)
            self.conv2_relu = tf.nn.relu(self.conv2_bn.bnorm)
            self.pool2 = tf.nn.max_pool(self.conv2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool2')
        
            self.conv3 = tf.nn.conv2d(self.pool2,self.conv3_w,strides=[1,1,1,1],padding="VALID",name = 'conv3') 
            self.conv3_bn = batch_norm(self.conv3,self.is_training,self.sess)
            self.conv3_relu = tf.nn.relu(self.conv3_bn.bnorm)    
        
            self.conv4 = tf.nn.conv2d(self.conv3,self.conv4_w,strides=[1,1,1,1],padding="VALID",name='conv4')
            self.conv4_bn = batch_norm(self.conv4,self.is_training,self.sess)
            self.conv4_relu = tf.nn.relu(self.conv4_bn.bnorm)    
        
            self.conv5 = tf.nn.conv2d(self.conv4,self.conv5_w,strides=[1,1,1,1],padding="VALID",name='conv5')
            self.conv5_bn = batch_norm(self.conv5,self.is_training,self.sess)
            self.conv5_relu = tf.nn.relu(self.conv5_bn.bnorm)
            self.pool3 = tf.nn.max_pool(self.conv5_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool3')
            
            
            #-----------------------------------refer input---------------------------------------------------------------
            self.conv1_ = tf.nn.conv2d(self.x_,self.conv1_w_,strides=[1,1,1,1],padding="VALID",name='conv1_') 
            self.conv1_bn_ = batch_norm(self.conv1_,self.is_training,self.sess)
            self.conv1_relu_ = tf.nn.relu(self.conv1_bn_.bnorm)
            self.pool1_ = tf.nn.max_pool(self.conv1_relu_,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool1_')   
        
            self.conv2_ = tf.nn.conv2d(self.pool1_,self.conv2_w_,strides=[1,1,1,1],padding="VALID",name='conv2_')
            self.conv2_bn_ = batch_norm(self.conv2_,self.is_training,self.sess)
            self.conv2_relu_ = tf.nn.relu(self.conv2_bn_.bnorm)
            self.pool2_ = tf.nn.max_pool(self.conv2_relu_,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool2_')
        
            self.conv3_ = tf.nn.conv2d(self.pool2_,self.conv3_w_,strides=[1,1,1,1],padding="VALID",name='conv3_') 
            self.conv3_bn_ = batch_norm(self.conv3_,self.is_training,self.sess)
            self.conv3_relu_ = tf.nn.relu(self.conv3_bn_.bnorm)    
        
            self.conv4_ = tf.nn.conv2d(self.conv3_,self.conv4_w_,strides=[1,1,1,1],padding="VALID",name='conv4_')
            self.conv4_bn_ = batch_norm(self.conv4_,self.is_training,self.sess)
            self.conv4_relu_ = tf.nn.relu(self.conv4_bn_.bnorm)    
        
            self.conv5_ = tf.nn.conv2d(self.conv4_,self.conv5_w_,strides=[1,1,1,1],padding="VALID",name='conv5_')
            self.conv5_bn_ = batch_norm(self.conv5_,self.is_training,self.sess)
            self.conv5_relu_ = tf.nn.relu(self.conv5_bn_.bnorm)
            self.pool3_ = tf.nn.max_pool(self.conv5_relu_,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool3_')
        
            self.concat=tf.concat(3,[self.pool3,self.pool3_])
        
            self.conv6 = tf.nn.conv2d(self.concat,self.conv6_w,strides=[1,1,1,1],padding="VALID",name='conv6') 
            self.conv6_bn = batch_norm(self.conv6,self.is_training,self.sess)
            self.conv6_relu = tf.nn.relu(self.conv6_bn.bnorm)   
        
            self.conv7 = tf.nn.conv2d(self.conv6,self.conv7_w,strides=[1,1,1,1],padding="VALID",name='conv7')
            self.conv7_bn = batch_norm(self.conv7,self.is_training,self.sess)
            self.conv7_relu = tf.nn.relu(self.conv7_bn.bnorm) 
            self.pool7 = tf.nn.avg_pool(self.conv7_relu, ksize=[1,3,3,1],strides=[1,3,3,1],padding="SAME",name='pool7' )
        
            self.conv8 = tf.nn.conv2d(self.pool7,self.conv8_w,strides=[1,1,1,1],padding = "SAME",name='conv8')
            self.conv8_bn = batch_norm(self.conv8,self.is_training,self.sess)
            self.conv8_relu = tf.nn.relu(self.conv8_bn.bnorm)
        
            self.conv9 = tf.nn.conv2d(self.conv8,self.conv9_w,strides=[1,1,1,1],padding = "VALID",name='conv9')
            self.conv9_bn = batch_norm(self.conv9,self.is_training,self.sess)
        
            self.pool9 = tf.nn.avg_pool(self.conv9_bn.bnorm, ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool9')
        
            self.pool9_flat = tf.reshape(self.pool9,[-1,3072])
            self.fc1 = tf.matmul(self.pool9_flat,self.fc1_w)+self.fc1_b
            self.fc1 = tf.nn.relu(self.fc1)       
            self.fc2 = tf.matmul(self.fc1,self.fc2_w)+self.fc2_b
            self.saver = tf.train.Saver({
             'conv1_w':self.conv1_w,   'conv2_w':self.conv2_w,   'conv3_w':self.conv3_w,   'conv4_w':self.conv4_w,   'conv5_w':self.conv5_w, 
             'conv1_w_':self.conv1_w_, 'conv2_w_':self.conv2_w_, 'conv3_w_':self.conv3_w_, 'conv4_w_':self.conv4_w_, 'conv5_w_':self.conv5_w_, 
             'conv6_w':self.conv6_w,   'conv7_w':self.conv7_w,   'conv8_w':self.conv8_w,   'conv9_w':self.conv9_w, 
             'fc1_w':self.fc1_w,       'fc1_b':self.fc1_b,       'fc2_w':self.fc2_w,       'fc2_b':self.fc2_b,
             'conv1_bn_scale' :self.conv1_bn.scale, 'conv1_bn_beta':self.conv1_bn.beta,  'conv1_bn_pop_mean' :self.conv1_bn.pop_mean, 'conv1_bn_pop_var' :self.conv1_bn.pop_var,
             'conv2_bn_scale' :self.conv2_bn.scale, 'conv2_bn_beta':self.conv2_bn.beta,  'conv2_bn_pop_mean' :self.conv2_bn.pop_mean, 'conv2_bn_pop_var' :self.conv2_bn.pop_var,
             'conv3_bn_scale' :self.conv3_bn.scale, 'conv3_bn_beta':self.conv3_bn.beta,  'conv3_bn_pop_mean' :self.conv3_bn.pop_mean, 'conv3_bn_pop_var' :self.conv3_bn.pop_var,
             'conv4_bn_scale' :self.conv4_bn.scale, 'conv4_bn_beta':self.conv4_bn.beta,  'conv4_bn_pop_mean' :self.conv4_bn.pop_mean, 'conv4_bn_pop_var' :self.conv4_bn.pop_var,
             'conv5_bn_scale' :self.conv5_bn.scale, 'conv5_bn_beta':self.conv5_bn.beta,  'conv5_bn_pop_mean' :self.conv5_bn.pop_mean, 'conv5_bn_pop_var' :self.conv5_bn.pop_var,
             'conv1_bn_scale_':self.conv1_bn_.scale,'conv1_bn_beta_':self.conv1_bn_.beta,'conv1_bn_pop_mean_':self.conv1_bn_.pop_mean,'conv1_bn_pop_var_':self.conv1_bn_.pop_var,
             'conv2_bn_scale_':self.conv2_bn_.scale,'conv2_bn_beta_':self.conv2_bn_.beta,'conv2_bn_pop_mean_':self.conv2_bn_.pop_mean,'conv2_bn_pop_var_':self.conv2_bn_.pop_var,
             'conv3_bn_scale_':self.conv3_bn_.scale,'conv3_bn_beta_':self.conv3_bn_.beta,'conv3_bn_pop_mean_':self.conv3_bn_.pop_mean,'conv3_bn_pop_var_':self.conv3_bn_.pop_var,
             'conv4_bn_scale_':self.conv4_bn_.scale,'conv4_bn_beta_':self.conv4_bn_.beta,'conv4_bn_pop_mean_':self.conv4_bn_.pop_mean,'conv4_bn_pop_var_':self.conv4_bn_.pop_var,
             'conv5_bn_scale_':self.conv5_bn_.scale,'conv5_bn_beta_':self.conv5_bn_.beta,'conv5_bn_pop_mean_':self.conv5_bn_.pop_mean,'conv5_bn_pop_var_':self.conv5_bn_.pop_var,
             'conv6_bn_scale' :self.conv6_bn.scale, 'conv6_bn_beta':self.conv6_bn.beta,  'conv6_bn_pop_mean' :self.conv6_bn.pop_mean, 'conv6_bn_pop_var' :self.conv6_bn.pop_var,
             'conv7_bn_scale' :self.conv7_bn.scale, 'conv7_bn_beta':self.conv7_bn.beta,  'conv7_bn_pop_mean' :self.conv7_bn.pop_mean, 'conv7_bn_pop_var' :self.conv7_bn.pop_var, 
             'conv8_bn_scale' :self.conv8_bn.scale, 'conv8_bn_beta':self.conv8_bn.beta,  'conv8_bn_pop_mean' :self.conv8_bn.pop_mean, 'conv8_bn_pop_var' :self.conv8_bn.pop_var,
             'conv9_bn_scale' :self.conv9_bn.scale, 'conv9_bn_beta':self.conv9_bn.beta,  'conv9_bn_pop_mean' :self.conv9_bn.pop_mean, 'conv9_bn_pop_var' :self.conv9_bn.pop_var })
 


            self.t_x = tf.placeholder(tf.float32, shape = [None,400,300,1])
            self.t_x_= tf.placeholder(tf.float32, shape = [None,400,300,1])
            self.t_is_training = tf.placeholder(tf.bool, [])
            
            self.t_conv1_w = tf.Variable(tf.random_normal(([3,3,1,96]),stddev=np.sqrt(2.0/9/1)),dtype="float32",name='conv1_w')
            self.t_conv2_w = tf.Variable(tf.random_normal(([3,3,96,256]),stddev=np.sqrt(2.0/9/96)),dtype="float32",name='conv2_w')
            self.t_conv3_w = tf.Variable(tf.random_normal(([3,3,256,384]),stddev=np.sqrt(2.0/9/256)),dtype="float32",name='conv3_w')
            self.t_conv4_w = tf.Variable(tf.random_normal(([3,3,384,384]),stddev=np.sqrt(2.0/9/384)),dtype="float32",name='conv4_w')
            self.t_conv5_w = tf.Variable(tf.random_normal(([3,3,384,256]),stddev=np.sqrt(2.0/9/384)),dtype="float32",name='conv5_w')
        
            self.t_conv1_w_ = tf.Variable(tf.random_normal(([3,3,1,96]),stddev=np.sqrt(2.0/9/1)),dtype="float32",name='conv1_w_')
            self.t_conv2_w_ = tf.Variable(tf.random_normal(([3,3,96,256]),stddev=np.sqrt(2.0/9/96)),dtype="float32",name='conv2_w_')
            self.t_conv3_w_ = tf.Variable(tf.random_normal(([3,3,256,384]),stddev=np.sqrt(2.0/9/256)),dtype="float32",name='conv3_w_')
            self.t_conv4_w_ = tf.Variable(tf.random_normal(([3,3,384,384]),stddev=np.sqrt(2.0/9/384)),dtype="float32",name='conv4_w_')
            self.t_conv5_w_ = tf.Variable(tf.random_normal(([3,3,384,256]),stddev=np.sqrt(2.0/9/384)),dtype="float32",name='conv5_w_')
        
        
            self.t_conv6_w = tf.Variable(tf.random_normal(([3,3,512,256]),stddev=np.sqrt(2.0/9/512)),dtype="float32",name='conv6_w')
            self.t_conv7_w = tf.Variable(tf.random_normal(([3,3,256,128]),stddev=np.sqrt(2.0/9/256)),dtype="float32",name='conv7_w')        
            self.t_conv8_w = tf.Variable(tf.random_normal(([3,3,128,128]),stddev=np.sqrt(2.0/9/128)),dtype="float32",name='conv8_w')        
            self.t_conv9_w = tf.Variable(tf.random_normal(([3,3,128,128]),stddev=np.sqrt(2.0/9/128)),dtype="float32",name='conv9_w')
        
            self.t_fc1_w = tf.Variable(tf.random_uniform([3072,64],-1/np.sqrt(3072),1/np.sqrt(3072)),name='fc1_w')
            self.t_fc1_b = tf.Variable(tf.random_uniform([64],0,1e-6),name='fc1_b')
        
            self.t_fc2_w = tf.Variable(tf.random_uniform([64,6],-1/np.sqrt(64),1/np.sqrt(64)),name='fc2_w')
            self.t_fc2_b = tf.Variable(tf.random_uniform([6],0,1e-4),name='fc2_b')
            #--------------------------------------current input-------------------------------------------------------       
        
            self.t_conv1 = tf.nn.conv2d(self.t_x,self.t_conv1_w,strides=[1,1,1,1],padding="VALID",name='conv1') 
            self.t_conv1_bn = batch_norm(self.t_conv1,self.t_is_training,self.sess,self.conv1_bn)
            self.t_conv1_relu = tf.nn.relu(self.t_conv1_bn.bnorm)
            self.t_pool1 = tf.nn.max_pool(self.t_conv1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")   
        
            self.t_conv2 = tf.nn.conv2d(self.t_pool1,self.t_conv2_w,strides=[1,1,1,1],padding="VALID",name='conv2')
            self.t_conv2_bn = batch_norm(self.t_conv2,self.t_is_training,self.sess,self.conv2_bn)
            self.t_conv2_relu = tf.nn.relu(self.t_conv2_bn.bnorm)
            self.t_pool2 = tf.nn.max_pool(self.t_conv2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool2')
        
            self.t_conv3 = tf.nn.conv2d(self.t_pool2,self.t_conv3_w,strides=[1,1,1,1],padding="VALID",name = 'conv3') 
            self.t_conv3_bn = batch_norm(self.t_conv3,self.t_is_training,self.sess,self.conv3_bn)
            self.t_conv3_relu = tf.nn.relu(self.t_conv3_bn.bnorm)    
        
            self.t_conv4 = tf.nn.conv2d(self.t_conv3,self.t_conv4_w,strides=[1,1,1,1],padding="VALID",name='conv4')
            self.t_conv4_bn = batch_norm(self.t_conv4,self.t_is_training,self.sess,self.conv4_bn)
            self.t_conv4_relu = tf.nn.relu(self.t_conv4_bn.bnorm)    
        
            self.t_conv5 = tf.nn.conv2d(self.t_conv4,self.t_conv5_w,strides=[1,1,1,1],padding="VALID",name='conv5')
            self.t_conv5_bn = batch_norm(self.t_conv5,self.t_is_training,self.sess,self.conv5_bn)
            self.t_conv5_relu = tf.nn.relu(self.t_conv5_bn.bnorm)
            self.t_pool3 = tf.nn.max_pool(self.t_conv5_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool3')
            
            
            #-----------------------------------refer input---------------------------------------------------------------
            self.t_conv1_ = tf.nn.conv2d(self.t_x_,self.t_conv1_w_,strides=[1,1,1,1],padding="VALID",name='conv1_') 
            self.t_conv1_bn_ = batch_norm(self.t_conv1_,self.t_is_training,self.sess,self.conv1_bn_)
            self.t_conv1_relu_ = tf.nn.relu(self.t_conv1_bn_.bnorm)
            self.t_pool1_ = tf.nn.max_pool(self.t_conv1_relu_,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool1_')   
        
            self.t_conv2_ = tf.nn.conv2d(self.t_pool1_,self.t_conv2_w_,strides=[1,1,1,1],padding="VALID",name='conv2_')
            self.t_conv2_bn_ = batch_norm(self.t_conv2_,self.t_is_training,self.sess,self.conv2_bn_)
            self.t_conv2_relu_ = tf.nn.relu(self.t_conv2_bn_.bnorm)
            self.t_pool2_ = tf.nn.max_pool(self.t_conv2_relu_,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool2_')
        
            self.t_conv3_ = tf.nn.conv2d(self.t_pool2_,self.t_conv3_w_,strides=[1,1,1,1],padding="VALID",name='conv3_') 
            self.t_conv3_bn_ = batch_norm(self.t_conv3_,self.t_is_training,self.sess,self.conv3_bn_)
            self.t_conv3_relu_ = tf.nn.relu(self.t_conv3_bn_.bnorm)    
        
            self.t_conv4_ = tf.nn.conv2d(self.t_conv3_,self.t_conv4_w_,strides=[1,1,1,1],padding="VALID",name='conv4_')
            self.t_conv4_bn_ = batch_norm(self.t_conv4_,self.t_is_training,self.sess,self.conv4_bn_)
            self.t_conv4_relu_ = tf.nn.relu(self.t_conv4_bn_.bnorm)    
        
            self.t_conv5_ = tf.nn.conv2d(self.t_conv4_,self.t_conv5_w_,strides=[1,1,1,1],padding="VALID",name='conv5_')
            self.t_conv5_bn_ = batch_norm(self.t_conv5_,self.t_is_training,self.sess,self.conv5_bn_)
            self.t_conv5_relu_ = tf.nn.relu(self.t_conv5_bn_.bnorm)
            self.t_pool3_ = tf.nn.max_pool(self.t_conv5_relu_,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool3_')
        
            self.t_concat=tf.concat(3,[self.t_pool3,self.t_pool3_])
        
            self.t_conv6 = tf.nn.conv2d(self.t_concat,self.t_conv6_w,strides=[1,1,1,1],padding="VALID",name='conv6') 
            self.t_conv6_bn = batch_norm(self.t_conv6,self.t_is_training,self.sess,self.conv6_bn)
            self.t_conv6_relu = tf.nn.relu(self.t_conv6_bn.bnorm)   
        
            self.t_conv7 = tf.nn.conv2d(self.t_conv6,self.t_conv7_w,strides=[1,1,1,1],padding="VALID",name='conv7')
            self.t_conv7_bn = batch_norm(self.t_conv7,self.t_is_training,self.sess,self.conv7_bn)
            self.t_conv7_relu = tf.nn.relu(self.t_conv7_bn.bnorm) 
            self.t_pool7 = tf.nn.avg_pool(self.t_conv7_relu, ksize=[1,3,3,1],strides=[1,3,3,1],padding="SAME",name='pool7' )
        
            self.t_conv8 = tf.nn.conv2d(self.t_pool7,self.t_conv8_w,strides=[1,1,1,1],padding = "SAME",name='conv8')
            self.t_conv8_bn = batch_norm(self.t_conv8,self.t_is_training,self.sess,self.conv8_bn)
            self.t_conv8_relu = tf.nn.relu(self.t_conv8_bn.bnorm)
        
            self.t_conv9 = tf.nn.conv2d(self.t_conv8,self.t_conv9_w,strides=[1,1,1,1],padding = "VALID",name='conv9')
            self.t_conv9_bn = batch_norm(self.t_conv9,self.t_is_training,self.sess,self.conv9_bn)
        
            self.t_pool9 = tf.nn.avg_pool(self.t_conv9_bn.bnorm, ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool9')
        
            self.t_pool9_flat = tf.reshape(self.t_pool9,[-1,3072])
            self.t_fc1 = tf.matmul(self.t_pool9_flat,self.t_fc1_w)+self.t_fc1_b
            self.t_fc1 = tf.nn.relu(self.t_fc1)       
            self.t_fc2 = tf.matmul(self.t_fc1,self.t_fc2_w)+self.t_fc2_b
            self.t_saver = tf.train.Saver({
             'conv1_w':self.t_conv1_w,   'conv2_w':self.t_conv2_w,   'conv3_w':self.t_conv3_w,   'conv4_w':self.t_conv4_w,   'conv5_w':self.t_conv5_w, 
             'conv1_w_':self.t_conv1_w_, 'conv2_w_':self.t_conv2_w_, 'conv3_w_':self.t_conv3_w_, 'conv4_w_':self.t_conv4_w_, 'conv5_w_':self.t_conv5_w_, 
             'conv6_w':self.t_conv6_w,   'conv7_w':self.t_conv7_w,   'conv8_w':self.t_conv8_w,   'conv9_w':self.t_conv9_w, 
             'fc1_w':self.t_fc1_w,       'fc1_b':self.t_fc1_b,       'fc2_w':self.t_fc2_w,       'fc2_b':self.t_fc2_b,
             'conv1_bn_scale' :self.t_conv1_bn.scale, 'conv1_bn_beta':self.t_conv1_bn.beta,  'conv1_bn_pop_mean' :self.t_conv1_bn.pop_mean, 'conv1_bn_pop_var' :self.t_conv1_bn.pop_var,
             'conv2_bn_scale' :self.t_conv2_bn.scale, 'conv2_bn_beta':self.t_conv2_bn.beta,  'conv2_bn_pop_mean' :self.t_conv2_bn.pop_mean, 'conv2_bn_pop_var' :self.t_conv2_bn.pop_var,
             'conv3_bn_scale' :self.t_conv3_bn.scale, 'conv3_bn_beta':self.t_conv3_bn.beta,  'conv3_bn_pop_mean' :self.t_conv3_bn.pop_mean, 'conv3_bn_pop_var' :self.t_conv3_bn.pop_var,
             'conv4_bn_scale' :self.t_conv4_bn.scale, 'conv4_bn_beta':self.t_conv4_bn.beta,  'conv4_bn_pop_mean' :self.t_conv4_bn.pop_mean, 'conv4_bn_pop_var' :self.t_conv4_bn.pop_var,
             'conv5_bn_scale' :self.t_conv5_bn.scale, 'conv5_bn_beta':self.t_conv5_bn.beta,  'conv5_bn_pop_mean' :self.t_conv5_bn.pop_mean, 'conv5_bn_pop_var' :self.t_conv5_bn.pop_var,
             'conv1_bn_scale_':self.t_conv1_bn_.scale,'conv1_bn_beta_':self.t_conv1_bn_.beta,'conv1_bn_pop_mean_':self.t_conv1_bn_.pop_mean,'conv1_bn_pop_var_':self.t_conv1_bn_.pop_var,
             'conv2_bn_scale_':self.t_conv2_bn_.scale,'conv2_bn_beta_':self.t_conv2_bn_.beta,'conv2_bn_pop_mean_':self.t_conv2_bn_.pop_mean,'conv2_bn_pop_var_':self.t_conv2_bn_.pop_var,
             'conv3_bn_scale_':self.t_conv3_bn_.scale,'conv3_bn_beta_':self.t_conv3_bn_.beta,'conv3_bn_pop_mean_':self.t_conv3_bn_.pop_mean,'conv3_bn_pop_var_':self.t_conv3_bn_.pop_var,
             'conv4_bn_scale_':self.t_conv4_bn_.scale,'conv4_bn_beta_':self.t_conv4_bn_.beta,'conv4_bn_pop_mean_':self.t_conv4_bn_.pop_mean,'conv4_bn_pop_var_':self.t_conv4_bn_.pop_var,
             'conv5_bn_scale_':self.t_conv5_bn_.scale,'conv5_bn_beta_':self.t_conv5_bn_.beta,'conv5_bn_pop_mean_':self.t_conv5_bn_.pop_mean,'conv5_bn_pop_var_':self.t_conv5_bn_.pop_var,
             'conv6_bn_scale' :self.t_conv6_bn.scale, 'conv6_bn_beta':self.t_conv6_bn.beta,  'conv6_bn_pop_mean' :self.t_conv6_bn.pop_mean, 'conv6_bn_pop_var' :self.t_conv6_bn.pop_var,
             'conv7_bn_scale' :self.t_conv7_bn.scale, 'conv7_bn_beta':self.t_conv7_bn.beta,  'conv7_bn_pop_mean' :self.t_conv7_bn.pop_mean, 'conv7_bn_pop_var' :self.t_conv7_bn.pop_var, 
             'conv8_bn_scale' :self.t_conv8_bn.scale, 'conv8_bn_beta':self.t_conv8_bn.beta,  'conv8_bn_pop_mean' :self.t_conv8_bn.pop_mean, 'conv8_bn_pop_var' :self.t_conv8_bn.pop_var,
             'conv9_bn_scale' :self.t_conv9_bn.scale, 'conv9_bn_beta':self.t_conv9_bn.beta,  'conv9_bn_pop_mean' :self.t_conv9_bn.pop_mean, 'conv9_bn_pop_var' :self.t_conv9_bn.pop_var })

            #cost of actor network:
            self.q_gradient_input = tf.placeholder("float",[None,6]) #gets input from action_gradient computed in critic network file
            self.actor_parameters = [self.conv1_w,self.conv2_w,self.conv3_w,self.conv4_w,self.conv5_w, self.conv1_w_,self.conv2_w_,self.conv3_w_,self.conv4_w_,\
                                     self.conv5_w_, self.conv6_w,self.conv7_w,self.conv8_w,self.conv9_w,self.fc1_w,self.fc1_b,self.fc2_w,self.fc2_b,\
                                     self.conv1_bn.scale,self.conv1_bn.beta,self.conv2_bn.scale,self.conv2_bn.beta,self.conv3_bn.scale,self.conv3_bn.beta,\
                                     self.conv4_bn.scale,self.conv4_bn.beta,self.conv5_bn.scale,self.conv5_bn.beta,self.conv1_bn_.scale,self.conv1_bn_.beta,\
                                     self.conv2_bn_.scale,self.conv2_bn_.beta,self.conv3_bn_.scale,self.conv3_bn_.beta,self.conv4_bn_.scale,self.conv4_bn_.beta,\
                                     self.conv5_bn_.scale,self.conv5_bn_.beta,self.conv6_bn.scale,self.conv6_bn.beta,self.conv7_bn.scale,self.conv7_bn.beta,\
                                     self.conv8_bn.scale,self.conv8_bn.beta,self.conv9_bn.scale,self.conv9_bn.beta ]

            self.parameters_gradients = tf.gradients(self.fc2,self.actor_parameters,-self.q_gradient_input)#/BATCH_SIZE) changed -self.q_gradient to -
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,epsilon=1e-08).apply_gradients(zip(self.parameters_gradients,self.actor_parameters))  
            #initialize all tensor variable parameters:
            #self.sess.run(tf.initialize_all_variables()) 
            #-----------------------------------------------------------------------------
            self.saver.restore(self.sess, model)
            self.t_saver.restore(self.sess, model)            
            
    def evaluate_actor(self,x,x_,is_training=False):
        return self.sess.run(self.fc2, feed_dict={self.x:x,self.x_:x_,self.is_training:is_training}) 
        
    def evaluate_target_actor(self,x,x_,is_training=False):
        return self.sess.run(self.t_fc2, feed_dict={self.x:x,self.x_:x_,self.is_training:is_training})
        
    def evaluate_actor_loss(self,x,x_,y,is_training=False):
        return self.sess.run(self.loss,feed_dict={self.x:x,self.x_:x_,self.y:y,self.is_training:is_training})
        
    def train_actor(self,x,x_,y,is_training=True):
        _,loss,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_=\
        self.sess.run([self.optimizer,self.loss,self.conv1_bn.train_mean,self.conv1_bn.train_var,self.conv2_bn.train_mean,self.conv2_bn.train_var,\
            self.conv3_bn.train_mean,self.conv3_bn.train_var,self.conv4_bn.train_mean, self.conv4_bn.train_var,self.conv5_bn.train_mean, self.conv5_bn.train_var,\
            self.conv1_bn_.train_mean,self.conv1_bn_.train_var,self.conv2_bn_.train_mean,self.conv2_bn_.train_var,self.conv3_bn_.train_mean,self.conv3_bn_.train_var,\
            self.conv4_bn_.train_mean,self.conv4_bn_.train_var,self.conv5_bn_.train_mean, self.conv5_bn_.train_var,self.conv6_bn.train_mean,self.conv6_bn.train_var,\
            self.conv7_bn.train_mean,self.conv7_bn.train_var,self.conv8_bn.train_mean, self.conv8_bn.train_var,self.conv9_bn.train_mean, self.conv9_bn.train_var],\
            feed_dict={ self.x: x,self.x_: x_, self.y: y,self.is_training: is_training})
        return loss
        
    def update_target_actor(self):
        self.sess.run([self.t_conv1_w.assign(tau*self.conv1_w+(1-tau)*self.t_conv1_w),self.t_conv2_w.assign(tau*self.conv2_w+(1-tau)*self.t_conv2_w),
                       self.t_conv3_w.assign(tau*self.conv3_w+(1-tau)*self.t_conv3_w),self.t_conv4_w.assign(tau*self.conv4_w+(1-tau)*self.t_conv4_w),
                       self.t_conv5_w.assign(tau*self.conv5_w+(1-tau)*self.t_conv5_w),self.t_conv1_w_.assign(tau*self.conv1_w_+(1-tau)*self.t_conv1_w_),
                       self.t_conv2_w_.assign(tau*self.conv2_w_+(1-tau)*self.t_conv2_w_),self.t_conv3_w_.assign(tau*self.conv3_w_+(1-tau)*self.t_conv3_w_),
                       self.t_conv4_w_.assign(tau*self.conv4_w_+(1-tau)*self.t_conv4_w_),self.t_conv5_w_.assign(tau*self.conv5_w_+(1-tau)*self.t_conv5_w_),
                       self.t_conv6_w.assign(tau*self.conv6_w+(1-tau)*self.t_conv6_w),self.t_conv7_w.assign(tau*self.conv7_w+(1-tau)*self.t_conv7_w),
                       self.t_conv8_w.assign(tau*self.conv8_w+(1-tau)*self.t_conv8_w),self.t_conv9_w.assign(tau*self.conv9_w+(1-tau)*self.t_conv9_w),
                       self.t_conv1_bn.updateTarget,self.t_conv2_bn.updateTarget,self.t_conv3_bn.updateTarget,self.t_conv4_bn.updateTarget,self.t_conv5_bn.updateTarget,
                       self.t_conv1_bn_.updateTarget,self.t_conv2_bn_.updateTarget,self.t_conv3_bn_.updateTarget,self.t_conv4_bn_.updateTarget,self.t_conv5_bn_.updateTarget,
                       self.t_conv6_bn.updateTarget,self.t_conv7_bn.updateTarget,self.t_conv8_bn.updateTarget,self.t_conv9_bn.updateTarget])
        
def main():
    #training data and test data prepare
    an = ActorNet()
    model_path='/home/wzl/design/pre_train_model_change_lr/'
    img_refer='/home/wzl/design/refer.tif'
    os.chdir('/home/wzl/design/img_data_mix_seq_train/')
    #training data
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
    ntrain = len(imgnames)

    #test data
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
    #refer image
    nbatch = 16 
    data_ = np.zeros([nbatch,300,400,1])
    img_origin = np.array(Image.open(img_refer))
    for i in range(nbatch):
        data_[i,]=img_origin[:,:,np.newaxis]
        i=i+1
    data_ = data_.transpose([0,2,1,3])

    training_epochs = 100    
    #saver.restore(sess,model_path+'061.ckpt')
    for epoch in range(training_epochs):
        for idx in range(int(ntrain/nbatch)):
            batchnames = imgnames[idx*nbatch:(idx+1)*nbatch]
            data = np.zeros([nbatch,300,400,1])
            i = 0
            for fn in batchnames:
                tempimg = np.array(Image.open(fn))
                data[i,] = tempimg[:,:,np.newaxis]
                i = i+1
            data = data.transpose([0,2,1,3])
            templabel = label[idx*nbatch:(idx+1)*nbatch]
            cost,learning_rate = an.train_actor(data,data_,templabel)
            index = epoch*750 + idx 
            if (index%20 ==0 and index>99):
                print ('Epoch:','%03d'%(epoch),'Loss:','%.5f'%(cost/16),'learning_rate:','%.9f'%(learning_rate))
                test_batchnames = []
                test_templabel = []
                randomdata = range(12000,12016)  
                randomlist = random.sample(randomdata,nbatch)
                for i in range(nbatch):
                    randomlist[i]=randomlist[i]-12000
                    test_batchnames.append(test_img_name[randomlist[i]])
                    test_templabel.append(testlabel[randomlist[i]])
                test_data = np.zeros([nbatch,300,400,1])
                j=0
                for test_fn in test_batchnames:
                    test_tempimg = np.array(Image.open(test_fn))
                    test_data[j,] = test_tempimg[:,:,np.newaxis]
                    j=j+1
                test_data=test_data.transpose([0,2,1,3])
                test_cost = an.evaluate_actor_loss(test_data,data_,test_templabel)      
                print (test_cost/16)
                print ('-----------------------------------')
                
        save_path = an.saver.save(an.sess,model_path+'%03d.ckpt'%(epoch))
        print("Model saved in file: %s" % save_path)

    print ('optimization finished!')

#if __name__ == '__main__':
#    main()    
