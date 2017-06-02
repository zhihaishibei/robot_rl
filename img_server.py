#!/usr/bin/env python
# coding=utf-8

'''receive image from client and send actor command'''

import socket
import numpy as np
from PIL import Image

class ImgServer(object):

    '''receive image from client and send actor command'''

    def __init__(self):
        '''init'''
        self.sock = socket.socket()
        self.host = ''
        self.port = 21567
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)

    def wait_for_connect(self):
        '''wait for client connection'''
        self.client, addr = self.sock.accept()
        print 'Got connection from', addr

    def receive_img(self):
        '''receive image from client'''
        line = self.client.recv(1024)
        with open('temp.tif','wb') as img:
            while(line):
                img.write(line)
                line = self.client.recv(1024)
        print 'Done receiving!'
        temp_img = np.array(Image.open('temp.tif'))
        return temp_img
    
    def send_actor_cmd(self, action_t):
        '''send actor command to robot'''
        str_buf = ''
        str_buf = str_buf + str(action_t[0, 0]) + ' '
        str_buf = str_buf + str(action_t[0, 1]) + ' '
        str_buf = str_buf + str(action_t[0, 2]) + ' '
        str_buf = str_buf + str(action_t[0, 3]) + ' '
        str_buf = str_buf + str(action_t[0, 4]) + ' '
        str_buf = str_buf + str(action_t[0, 5]) + ' '
        self.client.send(str_buf)

    def close_connect(self):
        self.client.close()


if __name__ == '__main__':
    img_server = ImgServer()
    img_server.wait_for_connect()
    while(True):
        temp_img = img_server.receive_img()
        img_server.close_connect()
        img_server.wait_for_connect()
