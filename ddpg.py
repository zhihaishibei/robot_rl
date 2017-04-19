import numpy as np
from rl_actor_class import ActorNet
from state_encoder import StateNet
from critic_net import CriticNet
from collections import deque
import random
from tensorflow_grad_inverter import grad_inverter
from PIL import Image
import pickle

#model path
actornet_pre_trained = ''
statenet_pre_trained = ''

REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA=0.99
is_grad_inverter = False
class DDPG:
    
    """ Deep Deterministic Policy Gradient Algorithm"""
    def __init__(self):
        self.actor_net = ActorNet(actornet_pre_trained)
        self.state_net = StateNet(statenet_pre_trained) 
        
        #Initialize Buffer Network:
        self.replay_memory = deque()
        
        #Intialize time step:
        self.time_step = 0
        self.counter = 0
        
        imgrefer = np.array(Image.open('refer.img')) 
        #refer data is used to infer actor 
        self.observe_refer_data = np.zeors(1,300,400,1)
        self.observe_refer_data[0,]=imgrefer[:,:,np.newaxis]
        self.observe_refer_data = self.observe_refer_data.transpose([0,2,1,3])
        self.observe_refer_datas = np.zeros(BATCH_SIZE,300,400,1)
        for i in range(BATCH_SIZE):
            self.observe_refer_datas[i,] = imgrefer[:,:,np.newaxis]
        self.observe_refer_datas = self.observe_refer_datas.transpose([0,2,1,3])
        
        
    # the type of x is [300,400]    
    def evaluate_actor(self, x):
        return self.actor_net.evaluate_actor(x,self.observe_refer_data)
    
    def add_experience(self, observation_1, observation_2, action, reward):
        self.observation_1 = observation_1
        self.observation_2 = observation_2
        self.action = action
        self.reward = reward
        self.done = done
        data = {'ot':a,'ot_1':b,'done':c,'action':d}
        output = open('data.pkl', 'wb')

        # Pickle dictionary using protocol 0.
        pickle.dump(data, output)
        output.close()
        self.replay_memory.append((self.observation_1, self.observation_2, self.action, self.reward,self.done))
        self.time_step = self.time_step + 1
        if(len(self.replay_memory)>REPLAY_MEMORY_SIZE):
            self.replay_memory.popleft()
            
        
    def minibatches(self):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        #state t and obeservation t
        self.observe_t_data_batch = [item[0] for item in batch]
        self.observe_t_data = np.zeros([BATCH_SIZE,300,400,1])
        self.state_t_data = np.zeros([BATCH_SIZE,128])
        
        for i in range(BATCH_SIZE):
            temp_state = self.state_net.evaluate_state(self.observe_t_data_batch[i])
            self.state_t_data[i] = temp_state
            observe_t_img = self.observe_t_data_batch[i][:,:,np.newaxis]
            self.observe_t_data[i,:,:,:] = observe_t_img

        self.observe_t_data  = self.observe_t_data.transpose([0,2,1,3])
        #state t+1 and observation t
        self.observe_t_1_data_batch = [item[1] for item in batch]
        self.observe_t_1_data = np.zeros([BATCH_SIZE,300,400,1])
        self.state_t_1_data = np.zeros([BATCH_SIZE,128])        
        for i in range(BATCH_SIZE):
            temp_state = self.state_net.evaluate_state(self.observe_t_1_data_batch[i])
            self.state_t_1_data[i] = temp_state
            observe_t_1_img = self.observe_data_batch_[i][:,:,np.newaxis]
            self.observe_t_1_data[i,:,:,:] = observe_t_1_img
        self.observe_t_1_data = self.observe_t_1_data.transpose([0,2,1,3])
        #action 
        self.action_batch = [item[2] for item in batch]
        self.action_batch = np.array(self.action_batch)
        self.action_batch = np.reshape(self.action_batch,[len(self.action_batch),-1])
        #reward 
        self.reward_batch = [item[3] for item in batch]
        self.reward_batch = np.array(self.reward_batch)
        #done
        self.done_batch = [item[4] for item in batch]
        self.done_batch = np.array(self.done_batch)  
                  
                 
    def train(self):
        #sample a random minibatch of N transitions from R
        self.minibatches()
        #action t1
        self.action_t_1_data = self.actor_net.evaluate_target_actor(self.observe_t_1_data,self.observe_refer_data)
        #Q'(s_i+1,a_i+1)        
        q_t_1 = self.critic_net.evaluate_target_critic(self.state_t_1_data,self.action_t_1_data) 
        self.y_i_batch=[]         
        for i in range(0,BATCH_SIZE):
                           
            if self.done_batch[i]:
                self.y_i_batch.append(self.reward_batch[i])
            else:
                
                self.y_i_batch.append(self.reward_batch[i] + GAMMA*q_t_1[i][0])                 
        
        self.y_i_batch=np.array(self.y_i_batch)
        self.y_i_batch = np.reshape(self.y_i_batch,[len(self.y_i_batch),1])
        
        # Update critic by minimizing the loss
        self.critic_net.train_critic(self.state_t_data, self.action_batch,self.y_i_batch)
        
        # Update actor proportional to the gradients:
        action_for_delQ = self.evaluate_actor(self.observe_t_data,self.observe_refer_data) 
        
        if is_grad_inverter:        
            self.del_Q_a = self.critic_net.compute_delQ_a(self.state_t_batch,action_for_delQ)#/BATCH_SIZE            
            self.del_Q_a = self.grad_inv.invert(self.del_Q_a,action_for_delQ) 
        else:
            self.del_Q_a = self.critic_net.compute_delQ_a(self.state_t_batch,action_for_delQ)[0]#/BATCH_SIZE
        
        # train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
        self.actor_net.train_actor(self.observe_t_data,self.observe_refer_datas,self.del_Q_a)
 
        # Update target Critic and actor network
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()
        
