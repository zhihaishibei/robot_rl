#!/usr/bin/env python
# coding=utf-8
'''implement the main function of the program'''

import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
from PIL import Image
# specify parameters here:
EPISODES = 10000
ACTION_DIM = 6
REFER_IMG = np.array(Image.open('refer.tif'))
ACTORNET_PRE_TRAINED = '/home/wzl/design/pre_train_model_change_lr/099.ckpt'
STATENET_PRE_TRAINED = '/home/wzl/design/encode_model_drop/099.ckpt'


def convert_img2data(img):
    '''convert [400,300] to [1, 300, 400, 1]'''
    observe_t_data = np.zeros(1, 300, 400, 1)
    observe_t_data[0, ] = img[:, :, np.newaxis]
    observe_t_data = observe_t_data.transpose([0, 2, 1, 3])
    return observe_t_data


def compute_cost(cur_data):
    '''compute cost function'''
    cost = np.sum(np.square(REFER_IMG - cur_data))
    return cost / 120000


def main():
    '''main function'''

    # Randomly initialize critic,actor,target critic, target actor network  and replay buffer
    agent = DDPG(ACTORNET_PRE_TRAINED, STATENET_PRE_TRAINED)
    exploration_noise = OUNoise(ACTION_DIM)
    # saving reward:
    reward_st = np.array([0])

    img_server = ImgServer()
    img_server.wait_for_connect()

    observe_t_img = img_server.receive_img()
    observe_t_data = convert_img2data(observe_t_img)
    actor_t = agent.evaluate_actor(observe_t_data)
    noise = exploration_noise.noise()
    actor_t = actor_t[0] + noise
    img_server.send_actor_cmd(actor_t)

    observe_t_1_img = observe_t_img
    actor_t_1 = actor_t
    img_server.close_connect()

    index = 1
    while True:
        observe_t_img = observe_t_1_img
        actor_t = actor_t_1
        img_server.wait_for_connect()
        observe_t_1_img = img_server.receive_img()
        observe_t_1_data = convert_img2data(observe_t_1_img)
        actor_t_1 = agent.evaluate_actor(observe_t_1_data)
        noise = exploration_noise.noise()
        actor_t_1 = actor_t_1[0] + noise
        cost = compute_cost(observe_t_img)
        agent.add_experience(observe_t_img, observe_t_1_img, actor_t, cost, index)
        if index > 32:
            agent.train()
        img_server.send_actor_cmd(actor_t_1)
        img_server.close_connect()

        index = index + 1


if __name__ == '__main__':
    main()
