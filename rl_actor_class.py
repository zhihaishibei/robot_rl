#!/usr/bin/env python

'''
used batch normalization
modify the routine to a class
has several class
used more space 
'''


import numpy as np
import tensorflow as tf
import os
from PIL import Image
  
class batch_norm:
    def __init__(self,inputs,is_training,is_input_data=False,parForTarget=None,decay = 0.95,TAU = 0.001,bn_param=None):
        
        if is_input_data:
            self.scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]),trainable=False,name=inputs.name.replace(':','')+'scale')
            self.beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),trainable=False,name=inputs.name.replace(':','')+'beta')
        else:
            self.scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]),name=inputs.name.replace(':','')+'scale')
            self.beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),name=inputs.name.replace(':','')+'beta')
        self.pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),trainable=False,name=inputs.name.replace(':','')+'pop_mean')
        self.pop_var = tf.Variable(tf.zeros([inputs.get_shape()[-1]]),trainable=False,name=inputs.name.replace(':','')+'pop_var')        
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
   
class regress_model:
    def __init__(self,x,x_,is_training):
        self.x = x
        self.x_ = x_
        self.is_training = is_training
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
    
        self.x = batch_norm(self.x,self.is_training,True)
        self.conv1 = tf.nn.conv2d(self.x.bnorm,self.conv1_w,strides=[1,1,1,1],padding="VALID",name='conv1') 
        self.conv1 = batch_norm(self.conv1,self.is_training)
        self.conv1 = tf.nn.relu(self.conv1.bnorm)
        self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")   
    
        self.conv2 = tf.nn.conv2d(self.pool1,self.conv2_w,strides=[1,1,1,1],padding="VALID",name='conv2')
        self.conv2 = batch_norm(self.conv2,self.is_training)
        self.conv2 = tf.nn.relu(self.conv2.bnorm)
        self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool2')
    
        self.conv3 = tf.nn.conv2d(self.pool2,self.conv3_w,strides=[1,1,1,1],padding="VALID",name = 'conv3') 
        self.conv3 = batch_norm(self.conv3,self.is_training)
        self.conv3 = tf.nn.relu(self.conv3.bnorm)    
    
        self.conv4 = tf.nn.conv2d(self.conv3,self.conv4_w,strides=[1,1,1,1],padding="VALID",name='conv4')
        self.conv4 = batch_norm(self.conv4,self.is_training)
        self.conv4 = tf.nn.relu(self.conv4.bnorm)    
    
        self.conv5 = tf.nn.conv2d(self.conv4,self.conv5_w,strides=[1,1,1,1],padding="VALID",name='conv5')
        self.conv5 = batch_norm(self.conv5,self.is_training)
        self.conv5 = tf.nn.relu(self.conv5.bnorm)
        self.pool3 = tf.nn.max_pool(self.conv5,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool3')
        #-----------------------------------refer input---------------------------------------------------------------
        self.x_ = batch_norm(self.x_,self.is_training,True)
        self.conv1_ = tf.nn.conv2d(self.x_.bnorm,self.conv1_w_,strides=[1,1,1,1],padding="VALID",name='conv1_') 
        self.conv1_ = batch_norm(self.conv1_,self.is_training)
        self.conv1_ = tf.nn.relu(self.conv1_.bnorm)
        self.pool1_ = tf.nn.max_pool(self.conv1_,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool1_')   
    
        self.conv2_ = tf.nn.conv2d(self.pool1_,self.conv2_w_,strides=[1,1,1,1],padding="VALID",name='conv2_')
        self.conv2_ = batch_norm(self.conv2_,self.is_training)
        self.conv2_ = tf.nn.relu(self.conv2_.bnorm)
        self.pool2_ = tf.nn.max_pool(self.conv2_,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool2_')
    
        self.conv3_ = tf.nn.conv2d(self.pool2_,self.conv3_w_,strides=[1,1,1,1],padding="VALID",name='conv3_') 
        self.conv3_ = batch_norm(self.conv3_,self.is_training)
        self.conv3_ = tf.nn.relu(self.conv3_.bnorm)    
    
        self.conv4_ = tf.nn.conv2d(self.conv3_,self.conv4_w_,strides=[1,1,1,1],padding="VALID",name='conv4_')
        self.conv4_ = batch_norm(self.conv4_,self.is_training)
        self.conv4_ = tf.nn.relu(self.conv4_.bnorm)    
    
        self.conv5_ = tf.nn.conv2d(self.conv4_,self.conv5_w_,strides=[1,1,1,1],padding="VALID",name='conv5_')
        self.conv5_ = batch_norm(self.conv5_,self.is_training)
        self.conv5_ = tf.nn.relu(self.conv5_.bnorm)
        self.pool3_ = tf.nn.max_pool(self.conv5_,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool3_')
    
        self.concat=tf.concat(3,[self.pool3,self.pool3_])
    
        self.conv6 = tf.nn.conv2d(self.concat,self.conv6_w,strides=[1,1,1,1],padding="VALID",name='conv6') 
        self.conv6 = batch_norm(self.conv6,self.is_training)
        self.conv6 = tf.nn.relu(self.conv6.bnorm)   
    
        self.conv7 = tf.nn.conv2d(self.conv6,self.conv7_w,strides=[1,1,1,1],padding="VALID",name='conv7')
        self.conv7 = batch_norm(self.conv7,self.is_training)
        self.conv7 = tf.nn.relu(self.conv7.bnorm) 
        self.pool7 = tf.nn.avg_pool(self.conv7, ksize=[1,3,3,1],strides=[1,3,3,1],padding="SAME",name='pool7' )
    
        self.conv8 = tf.nn.conv2d(self.pool7,self.conv8_w,strides=[1,1,1,1],padding = "SAME",name='conv8')
        self.conv8 = batch_norm(self.conv8,self.is_training)
        self.conv8 = tf.nn.relu(self.conv8.bnorm)
    
        self.conv9 = tf.nn.conv2d(self.conv8,self.conv9_w,strides=[1,1,1,1],padding = "VALID",name='conv9')
        self.conv9 = batch_norm(self.conv9,self.is_training)
    
        self.pool9 = tf.nn.avg_pool(self.conv9.bnorm, ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",name='pool9')
    
        self.pool9_flat = tf.reshape(self.pool9,[-1,3072])
        self.fc1 = tf.matmul(self.pool9_flat,self.fc1_w)+self.fc1_b
        self.fc1 = tf.nn.relu(self.fc1)       
        self.fc2 = tf.matmul(self.fc1,self.fc2_w)+self.fc2_b
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        
     
class ActorNet:
    def __init__(self):  
        self.actor_net = None 
        self.t_actor_net = None       
        #current net
        self.actor_model = tf.Graph()
        with self.actor_model.as_default():         
            self.x = tf.placeholder(tf.float32, shape = [None,400,300,1])
            self.x_= tf.placeholder(tf.float32, shape = [None,400,300,1])
            self.y = tf.placeholder(tf.float32, shape = [None,6])
            self.is_training = tf.placeholder(tf.bool, shape = [])
            self.actor_net = regress_model(self.x,self.x_,self.is_training)
            self.batch = tf.Variable(0, dtype="float", trainable=False)
            self.loss = tf.reduce_sum(tf.pow(self.y-self.actor_net.fc2,2))
            self.learning_rate = tf.train.exponential_decay(
                                0.001,                # Base learning rate.
                                self.batch * 16,  # Current index into the dataset.
                                12000,          # Decay step.
                                0.9,                # Decay rate.
                                staircase=True)

            self.train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)         

        self.sess = tf.Session(graph=self.actor_model)
        with self.sess.as_default():
            with self.actor_model.as_default():
                tf.global_variables_initializer().run()
        # target net
        self.t_actor_model = tf.Graph()
        self.t_sess = tf.Session(graph=self.t_actor_model)
        with self.t_actor_model.as_default():         
            self.t_x = tf.placeholder(tf.float32, shape = [None,400,300,1])
            self.t_x_= tf.placeholder(tf.float32, shape = [None,400,300,1])
            self.t_y = tf.placeholder(tf.float32, shape = [None,6])
            self.t_is_training = tf.placeholder(tf.bool, shape = [])
            self.t_actor_net = regress_model(self.t_x,self.t_x_,self.t_is_training)
            self.t_sess.run(self.t_actor_net.init) 

        #self.actor_net.saver.restore(self.sess,model_saved)

        #self.t_actor_net.saver.restore(self.t_sess, model_saved)  
        
 
     
    def evaluate_actor(self,x,x_):
        return self.sess.run(self.actor_net.fc2, feed_dict= {self.x:x,self.x_:x_,self.is_training:False})
        

    def evaluate_target_actor(self,x,x_):      
        return self.sess.run(self.t_actor_net.fc2, feed_dict= {self.x:x,self.x_:x_,self.is_training:False})  
        
        
    def train_actor(self,x,x_,y,save_model = False):
        _,loss = self.sess.run([self.train_step,self.loss], feed_dict={ self.x:x, self.x_:x_,self.y:y,self.is_training:True})
        if save_model:
            self.actor_net.saver.save(sess,'model.ckpt')
        return loss
        #self.actor_net.saver.save(self.sess,"temp.ckpt")
    def update_target_actor(self):
        self.t_saver.restore(self.t_sess,"temp.ckpt")


def main():
    #training data and test data prepare
    an = ActorNet()
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
    nbatch = 8 
    data_ = np.zeros([nbatch,300,400,1])
    img_origin = np.array(Image.open(img_refer))
    for i in range(nbatch):
        data_[i,]=img_origin[:,:,np.newaxis]
        i=i+1
    data_ = data_.transpose([0,2,1,3])

    training_epochs = 100    
    an.sess.run(an.actor_net.init)
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
            loss = an.train_actor(data,data_,templabel)
            print ('Loss=%.5f'%(loss))
#            if idx%20 ==0:
#                print ('Epoch:','%03d'%(epoch+1),'cost=','{:.5f}'.format(c),'LR=%.6f'%(lr))  
#                test_batchnames = []
#                test_templabel = []
#                randomdata = range(12000,12016)  
#                randomlist = random.sample(randomdata,nbatch)
#                for i in range(nbatch):
#                    randomlist[i]=randomlist[i]-12000
#                    test_batchnames.append(test_img_name[randomlist[i]])
#                    test_templabel.append(testlabel[randomlist[i]])
#                test_data = np.zeros([nbatch,height,width,channel])
#                j=0
#                for test_fn in test_batchnames:
#                    test_tempimg = np.array(Image.open(test_fn))
#                    test_data[j,] = test_tempimg[:,:,np.newaxis]
#                    j=j+1
#                test_data=test_data.transpose([0,2,1,3])
#                test_cost, summary= sess.run([test_loss, merged], feed_dict={x:test_data,x_:data_, y:test_templabel})
#                test_writer.add_summary(summary, idx)
#		print (test_cost)
#		print ('-----------------------------------')
#                
#        save_path = saver.save(sess,model_path+'%03d.ckpt'%(epoch))
#  	print("Model saved in file: %s" % save_path)
#
#    print ('optimization finished!')
#
#    referimg = np.array(Image.open('../refer.tif'))
#    referimg = referimg[np.newaxis,:,:,np,newaxis]
#    test_refer = referimg.transpose([0,2,1,3])
#
#    print("build successfully!")
#    for fn in test_img_name:
#        tempimg=np.array(Image.open(fn))
#        tempimg = tempimg[np.newaxis,:,:,np.newaxis]
#        test_data = tempimg.transpose([0,2,1,3])
#        test_pred = an.evaluate_actor(test_data,test_refer)
#        print test_pred[0]
#    
if __name__ == '__main__':
    main()    
