#!/usr/bin/env python

'''
decay = 0.9
used batch normalization
don't use spatial softmax
add tensorboard function to show realtime result
training result is saved in model_tensorboard
add name for each variable
log is in this path

modify the routine to a class

NOTE: the path of result and log must be absolute path or the result will not be saved

'''


import numpy as np
import tensorflow as tf
import os
from PIL import Image 
   
class StateNet:
    def __init__(self):
        tf.reset_default_graph()
        self.g=tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            #state network model parameters:
            self.x = tf.placeholder(tf.float32, shape = [None,400,300,1])
            batch_size = tf.shape(self.x)[0]
            self.is_training = tf.placeholder(tf.bool, [])
            
            #encoder parameter
            self.conv1_w = tf.Variable(tf.random_normal(([7,7,1,64]),stddev=np.sqrt(2.0/49/1)),dtype="float32",name='conv1_w')
            self.conv1_b = tf.Variable(tf.random_uniform([64],0,1e-6),name='conv1_b')
            self.conv2_w = tf.Variable(tf.random_normal(([5,5,64,32]),stddev=np.sqrt(2.0/25/64)),dtype="float32",name='conv2_w')
            self.conv2_b = tf.Variable(tf.random_uniform([32],0,1e-6),name='conv2_b')
            self.conv3_w = tf.Variable(tf.random_normal(([5,5,32,16]),stddev=np.sqrt(2.0/25/32)),dtype="float32",name='conv3_w')
            self.conv3_b = tf.Variable(tf.random_uniform([16],0,1e-6),name='conv3_b')
            self.conv4_w = tf.Variable(tf.random_normal(([3,3,16,1]),stddev=np.sqrt(2.0/9/16)),dtype="float32",name='conv4_w')
            self.conv4_b = tf.Variable(tf.random_uniform([1],0,1e-6),name='conv4_b')
        
            self.fc1_w = tf.Variable(tf.random_uniform([1900,128],-1/np.sqrt(1900),1/np.sqrt(1900)),name='fc1_w')
            self.fc1_b = tf.Variable(tf.random_uniform([128],0,1e-6),name='fc1_b')
        
            self.fc2_w = tf.Variable(tf.random_uniform([128,128],-1/np.sqrt(128),1/np.sqrt(128)),name='fc2_w')
            self.fc2_b = tf.Variable(tf.random_uniform([128],0,1e-4),name='fc2_b')
            #decoder parameter
            self.fc2_w_d = tf.Variable(tf.random_uniform([128,128],-1/np.sqrt(128),1/np.sqrt(128)),name='fc2_w_d')
            self.fc2_b_d = tf.Variable(tf.random_uniform([128],0,1e-6),name='fc2_b_d')
            self.fc1_w_d = tf.Variable(tf.random_uniform([128,1900],-1/np.sqrt(128),1/np.sqrt(128)),name='fc1_w_d')
            self.fc1_b_d = tf.Variable(tf.random_uniform([1900],0,1e-4),name='fc1_b_d')            
            self.conv4_w_d = tf.Variable(tf.random_normal(([3,3,16,1]),stddev=np.sqrt(2.0/9/1)),dtype="float32",name='conv4_w_d')
            self.conv4_b_d = tf.Variable(tf.random_uniform([16],0,1e-6),name='conv4_b_d')
            self.conv3_w_d = tf.Variable(tf.random_normal(([5,5,32,16]),stddev=np.sqrt(2.0/25/16)),dtype="float32",name='conv3_w_d')
            self.conv3_b_d = tf.Variable(tf.random_uniform([32],0,1e-6),name='conv3_b_d')
            self.conv2_w_d = tf.Variable(tf.random_normal(([5,5,64,32]),stddev=np.sqrt(2.0/25/32)),dtype="float32",name='conv2_w_d')
            self.conv2_b_d = tf.Variable(tf.random_uniform([64],0,1e-6),name='conv2_b_d')
            self.conv1_w_d = tf.Variable(tf.random_normal(([7,7,1,64]),stddev=np.sqrt(2.0/49/64)),dtype="float32",name='conv1_w_d')
            self.conv1_b_d = tf.Variable(tf.constant(0.0,shape=[1]),name='conv1_b_d')
            #encoder
            self.conv1 = tf.nn.conv2d(self.x,self.conv1_w,strides=[1,2,2,1],padding="SAME",name='conv1') 
            self.conv1_sig = tf.nn.relu(self.conv1+self.conv1_b)
            self.conv2 = tf.nn.conv2d(self.conv1_sig,self.conv2_w,strides=[1,2,2,1],padding="SAME",name='conv2')
            self.conv2_sig = tf.nn.relu(self.conv2+self.conv2_b)
            self.conv3 = tf.nn.conv2d(self.conv2_sig,self.conv3_w,strides=[1,2,2,1],padding="SAME",name = 'conv3') 
            self.conv3_sig = tf.nn.relu(self.conv3+self.conv3_b)
            self.conv4 = tf.nn.conv2d(self.conv3_sig,self.conv4_w,strides=[1,1,1,1],padding="SAME",name = 'conv4') 
            self.conv4_sig = tf.nn.relu(self.conv4+self.conv4_b)
            self.conv4_sig_flat = tf.reshape(self.conv4_sig,[-1,1900])
            self.fc1 = tf.nn.relu(tf.matmul(self.conv4_sig_flat,self.fc1_w)+self.fc1_b)
            def train_drop():
                return tf.nn.dropout(self.fc1,0.6)
            def test_without_drop():
                return self.fc1
            self.fc1 = tf.cond(self.is_training,train_drop,test_without_drop)
            self.encode = tf.nn.relu(tf.matmul(self.fc1, self.fc2_w) + self.fc2_b)
            #decoder
            self.fc2_d = tf.nn.relu(tf.matmul(self.encode,self.fc2_w_d)+self.fc2_b_d)
            self.fc1_d = tf.nn.relu(tf.matmul(self.fc2_d,self.fc1_w_d) + self.fc1_b_d)
            self.conv4_sig_flat_d = tf.reshape(self.fc1_d,(-1,50,38,1))
            self.conv4_d = tf.nn.conv2d_transpose(self.conv4_sig_flat_d,self.conv4_w_d,output_shape=(batch_size,50,38,16),strides=(1,1,1,1),padding='SAME')
            self.conv4_sig_d = tf.nn.relu(self.conv4_d + self.conv4_b_d)
            self.conv3_d = tf.nn.conv2d_transpose(self.conv4_sig_d,self.conv3_w_d,output_shape=(batch_size,100,75,32),strides=(1,2,2,1),padding='SAME')
            self.conv3_sig_d = tf.nn.relu(self.conv3_d + self.conv3_b_d)
            self.conv2_d = tf.nn.conv2d_transpose(self.conv3_sig_d,self.conv2_w_d,output_shape=(batch_size,200,150,64),strides=(1,2,2,1),padding='SAME')
            self.conv2_sig_d = tf.nn.relu(self.conv2_d + self.conv2_b_d)
            self.conv1_d= tf.nn.conv2d_transpose(self.conv2_sig_d,self.conv1_w_d,output_shape=(batch_size,400,300,1),strides=(1,2,2,1),padding='SAME')
            self.decode = tf.nn.relu(self.conv1_d + self.conv1_b_d)
      
            self.saver = tf.train.Saver({
             'conv1_w':self.conv1_w,  'conv1_b':self.conv1_b, 'conv2_w':self.conv2_w,  'conv2_b':self.conv2_b, 'conv3_w':self.conv3_w,  'conv3_b':self.conv3_b,
             'conv4_w':self.conv4_w,  'conv4_b':self.conv4_b, 'fc1_w':self.fc1_w, 'fc2_w':self.fc2_w, 'fc1_b':self.fc1_b, 'fc2_b':self.fc2_b, 
             'conv1_w_d':self.conv1_w_d,  'conv1_b_d':self.conv1_b_d, 'conv2_w_d':self.conv2_w_d,  'conv2_b_d':self.conv2_b_d, 'conv3_w_d':self.conv3_w_d, 
             'conv3_b_d':self.conv3_b_d,'conv4_w_d':self.conv4_w_d,  'conv4_b_d':self.conv4_b_d, 'fc1_w_d':self.fc1_w_d, 'fc2_w_d':self.fc2_w_d,
             'fc1_b_d':self.fc1_b_d, 'fc2_b_d':self.fc2_b_d })

            self.loss = tf.reduce_mean(tf.square(self.x-self.decode))
            self.batch = tf.Variable(0, dtype="float", trainable=False)
            self.learning_rate = tf.train.exponential_decay(
              0.001,                # Base learning rate.
              self.batch * 32,  # Current index into the dataset.
              12000,          # Decay step.
              0.9,                # Decay rate.
              staircase=True)
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step = self.batch)
            self.sess.run(tf.global_variables_initializer())
            
            
    def evaluate_state(self,x,is_training=False):
        return self.sess.run(self.decode, feed_dict={self.x:x,self.is_training:is_training}) 
        
    def evaluate_state_loss(self,x,is_training=False):
        return self.sess.run(self.loss,feed_dict={self.x:x,self.is_training:is_training})
        
    def train_state(self,x,y,is_training=True):
        _,loss,learning_rate = self.sess.run([self.train_step,self.loss,self.learning_rate], feed_dict={ self.x: x,self.is_training: is_training})
        return loss,learning_rate
            
   
        
def main():
    #training data and test data prepare
    an = StateNet()
    model_path='/home/wzl/design/encode_model_drop/'
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

    training_epochs = 100    
    #saver.restore(sess,model_path+'061.ckpt')
    for epoch in range(training_epochs):
        for idx in range(int(ntrain/nbatch)):
            batchnames = imgnames[idx*nbatch:(idx+1)*nbatch]
            data = np.zeros([nbatch,300,400,1])
            i = 0
            for fn in batchnames:
                tempimg = np.array(Image.open(fn))
                tempimg = tempimg
                data[i,] = tempimg[:,:,np.newaxis]
                i = i+1
            data = data.transpose([0,2,1,3])
            templabel = label[idx*nbatch:(idx+1)*nbatch]
            cost,LR = an.train_state(data,templabel)
            #index = epoch*750 + idx 
            #if (index%100 ==0 and index>99):
        print ('Epoch:','%03d'%(epoch),'Loss:','%.11f'%(cost),'LR:','%.9f'%(LR))

        tempimg_ = np.array(Image.open(imgnames[0]))
        data_ = np.zeros([1,300,400,1])
        data_ = tempimg_[np.newaxis,:,:,np.newaxis]
        data_ = data_.transpose([0,2,1,3])
        res_ = an.evaluate_state(data_)
        res_ = res_.transpose([0,2,1,3])
        res_ = np.squeeze(res_,[0,3])
        print res_
        res_ = np.round(res_)
        res_ = np.uint8(res_)
        print res_
        for i in range(300):
            for j in range(400):
                if(res_[i][j]>255):
                    res_[i][j]=255
        img_ = Image.fromarray(res_,'L')
        img_.save('../my.png')
                
        save_path = an.saver.save(an.sess,model_path+'%03d.ckpt'%(epoch))
        print("Model saved in file: %s" % save_path)

    print ('optimization finished!')
    
#if __name__ == '__main__':
#    main()    
