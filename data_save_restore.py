#!/usr/bin/env python
'''implement the function save and restore data'''
import numpy as np
import pickle
from PIL import Image

# saving the data


class DataSaveRestore(object):

    '''the class used to save and restore data from file'''
    def __init__(self):
        self.state_t = []
        self.state_t_1 = []
        self.actor = []
        self.reward = []

    def save_data(self, state_t, state_t_1, actor, reward, data_name):
        ''' used to save data to file'''
        dsr = DataSaveRestore()
        dsr.state_t = state_t
        dsr.state_t_1 = state_t_1
        dsr.actor = actor
        dsr.reward = reward
        with open(data_name, "wb") as my_save_data:
            pickle.dump(dsr, my_save_data)

    def restore_data(self, data_name):
        ''' restore the data from file'''
        with open(data_name, 'rb') as my_restore_data:
            return pickle.load(my_restore_data)


if __name__ == '__main__':

    print 'starting'
    tempimg = np.array(Image.open('00000.tif'))
    # print tempimg
    # img = tempimg
    # act = [1, 2, 3, 4.0]
    # rewar = 1
    # dsr = DataSaveRestore()
    # dsr.save_data(tempimg, img, act, rewar, 'test.pkl')
    # print 'saved data success!'
    # raw_input()
    # with open('test.pkl', 'rb') as my_restore_data:
    #     data = pickle.load(my_restore_data)
    # data = restore_data('mydata.pkl')

    dsr = DataSaveRestore()
    data = dsr.restore_data('test.pkl')
    # print data.reward
    # res = np.uint8(data.state_t)
    # img = Image.fromarray(res, 'L')
    # img.save('my1.png')
    # dsr.state_t = np.array(dsr.state_t_1)
    print data.state_t
