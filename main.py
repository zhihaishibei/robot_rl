#Implementation of Deep Deterministic Gradient with Tensor Flow"
# Author: Steven Spielberg Pon Kumar (github.com/stevenpjg)

import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
from PIL import Image


x_ = np.array(Image.open('refer.tif'))
#the type of x and x_ is [300,400] 
def compute_reward(x):
    return np.square(x-x_)/12000

data = dict{x_t,action,x_t_1,reward}

def main():
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    agent = DDPG()
    exploration_noise = OUNoise(env.action_space.shape[0])
    counter=0
    reward_per_episode = 0    
    total_reward=0
    #saving reward:
    reward_st = np.array([0])
      
    # network setup
    s = socket.socket()         # Create a socket object
    #host = socket.gethostname() # Get local machine name
    host = ''                    # Get local machine name
    port = 21567  # Reserve a port for your service.
    s.bind((host, port))
    
    s.listen(5)
    imgorigin_t = np.zeros(300,400)
    imgorigin_t_1 = np.zeros(300,400)
    actor_t = np.zeros(6)
    actor_t_1 =np.zeros(6)
    index = 0


    #the first time
    c, addr = s.accept()     # Establish connection with client.
    print ('Got connection from'), addr
    print ("Receiving...")
    l = c.recv(1024)
    f = open('temp.tif','wb')
    while (l):
        f.write(l)
        l = c.recv(1024)
    f.close()
    print ("Done Receiving")
    imgorigin_t = np.array(Image.open('temp.tif'))
    tempimg = imgorigin_t[np.newaxis,:,:,np.newaxis]
    tempimg = tempimg.transpose([0,2,1,3])
    test_pred = agent.evaluate_actor(tempimg)
    action_t = test_pred[0]

    print action_t

    str_buf = ''
    str_buf = str_buf+str(action_t[0,0])+" "
    str_buf = str_buf+str(action_t[0,1])+" "
    str_buf = str_buf+str(action_t[0,2])+" "
    str_buf = str_buf+str(action_t[0,3])+" "
    str_buf = str_buf+str(action_t[0,4])+" "
    str_buf = str_buf+str(action_t[0,5])+" "
    
    imgorigin_t_1 = imgorigin_t
    actor_t_1 = actor_t

    c.send(str_buf)
    c.close()
    
    index =1
    while True:
        #update imgorigin_t and actor_t
        imgorigin_t = img_origin_t_1
        actor_t = actor_t_1
        c, addr = s.accept()     # Establish connection with client.
        print ('Got connection from'), addr
        print ("Receiving...")
        l = c.recv(1024)
        f = open('temp.tif','wb')
        while (l):
            f.write(l)
            l = c.recv(1024)
        f.close()
        print ("Done Receiving")
        imgorigin_t_1 = np.array(Image.open('temp.tif'))
        tempimg = imgorigin_t_1[np.newaxis,:,:,np.newaxis]
        tempimg = tempimg.transpose([0,2,1,3])
        test_pred = agent.evaluate_actor(tempimg)
        action_t_1 = test_pred[0]
        print action_t_1

        reward = compute_reward(imgorigin_t_1)
        agent.add_experience(imgorigin_t,imgorigin_t_1,action_t,reward,index)

        if index > 32:
            agent.train()

        str_buf = ''
        str_buf = str_buf+str(action_t_1[0,0])+" "
        str_buf = str_buf+str(action_t_1[0,1])+" "
        str_buf = str_buf+str(action_t_1[0,2])+" "
        str_buf = str_buf+str(action_t_1[0,3])+" "
        str_buf = str_buf+str(action_t_1[0,4])+" "
        str_buf = str_buf+str(action_t_1[0,5])+" "
        c.send(str_buf)
        print("send action finished!")
        c.close()

        index = index+1



if __name__ == '__main__':
    main()    
