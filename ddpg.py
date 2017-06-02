#!/usr/bin/env python

'''the clas of ddpg'''

import numpy as np
from rl_actor_class import ActorNet
from state_encoder import StateNet
from critic_net import CriticNet
from collections import deque
import random
from tensorflow_grad_inverter import grad_inverter
from PIL import Image
import pickle
import data_save_restore

# model path
ACTORNET_PRE_TRAINED = ''
STATENET_PRE_TRAINED = ''
IMG_REFER = ''

REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
IS_GRAD_INVERTER = False


class DDPG:

    """ Deep Deterministic Policy Gradient Algorithm"""

    def __init__(self):
        self.actor_net = ActorNet(ACTORNET_PRE_TRAINED)
        self.state_net = StateNet(STATENET_PRE_TRAINED)

        # initialize the nember of data have been saved
        self.data_size = 0
        imgrefer = np.array(Image.open(IMG_REFER))
        # refer data is used to infer actor
        self.observe_refer_data = np.zeors(1, 300, 400, 1)
        self.observe_refer_data[0, ] = imgrefer[:, :, np.newaxis]
        self.observe_refer_data = self.observe_refer_data.transpose([0, 2, 1, 3])
        # refer datas is used to train
        self.observe_refer_datas = np.zeros(BATCH_SIZE, 300, 400, 1)
        for i in range(BATCH_SIZE):
            self.observe_refer_datas[i, ] = imgrefer[:, :, np.newaxis]
        self.observe_refer_datas = self.observe_refer_datas.transpose([0, 2, 1, 3])

    # the type of x is [1, 400, 300, 1]
    def evaluate_actor(self, x):
        return self.actor_net.evaluate_actor(x, self.observe_refer_data)

    # the type of
    def add_experience(self, observation_1, observation_2, action, reward, file_name):
        dsr = DataSaveRestore() 
        dsr.save_data(observation_1, observation_2, action, reward, file_name)
        self.data_size = self.data_size + 1

    # return a minibatch for train 
    def minibatches(self):
        dsr = DataSaveRestore() 
        self.observe_t_data_batch = np.zeros([BATCH_SIZE, 300, 400, 1])
        self.observe_t_1_data_batch = np.zeros([BATCH_SIZE, 300, 400, 1])
        self.state_t_data = np.zeros([BATCH_SIZE, 128])
        self.state_t_1_data = np.zeros([BATCH_SIZE, 128])
        self.action_batch = np.zeros([BATCH_SIZE, 6])
        self.reward_batch = np.zeros([BATCH_SIZE, 1])
        data_range = []

        if self.data_size < REPLAY_MEMORY_SIZE:
            data_range = self.__random_without_same(0, self.data_size, BATCH_SIZE)
        else:
            data_range = self.__random_without_same(self.data_size - REPLAY_MEMORY_SIZE)

        # state t,t1, and obeservation t,t1
        for (i, j) in zip(data_range, BATCH_SIZE):
            temp_data = dsr.restore_data('%d.pkl'%(i))
            temp_data_observe_t = temp_data.state_t
            temp_data_observe_t_1 = temp_data.state_t_1
            self.observe_t_data_batch[j, :, :, :] = temp_data_observe_t[:, :, np.newaxis] 
            self.observe_t_1_data_batch[j, :, :, :] = temp_data_observe_t_1[:, :, np.newaxis]
            self.action_batch[i, :] = temp_data.actor
            self.reward_batch[i, :] = temp_data.reward

        self.observe_t_data_batch = self.observe_t_data_batch.transpose([0, 2, 1, 3])
        self.state_t_data = self.state_net.evaluate_state(self.observe_t_data_batch)
        self.observe_t_1_data_batch = self.observe_t_1_data_batch.transpose([0, 2, 1, 3])
        self.state_t_1_data = self.state_net.evaluate_state(self.observe_t_1_data_batch)

    def train(self):
        # sample a random minibatch of N transitions from R
        self.minibatches()
        # action t1
        self.action_t_1_data = self.actor_net.evaluate_target_actor(
            self.observe_t_1_data_batch, self.observe_refer_data)
        # Q'(s_i+1,a_i+1)
        q_t_1 = self.critic_net.evaluate_target_critic(
            self.state_t_1_data, self.action_t_1_data)
        self.y_i_batch = []
        for i in range(0, BATCH_SIZE):
            self.y_i_batch.append( self.reward_batch[i] + GAMMA * q_t_1[i][0])

        self.y_i_batch = np.array(self.y_i_batch)
        self.y_i_batch = np.reshape(self.y_i_batch, [len(self.y_i_batch), 1])

        # Update critic by minimizing the loss
        self.critic_net.train_critic(
            self.state_t_data, self.action_batch, self.y_i_batch)

        # Update actor proportional to the gradients:
        action_for_delQ = self.evaluate_actor(
            self.observe_t_data_batch, self.observe_refer_data)

        self.del_Q_a = self.critic_net.compute_delQ_a(
            self.observe_t_data_batch, self.observe_refer_datas,  action_for_delQ)[0]  # /BATCH_SIZE

        # train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
        self.actor_net.train_actor(
            self.observe_t_data, self.observe_refer_datas, self.del_Q_a)

        # Update target Critic and actor network
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()

    def __random_without_same(self, mi, ma, num):
        temp = range(mi, ma)
        random.shuffle(temp)
        return temp[0:num]
