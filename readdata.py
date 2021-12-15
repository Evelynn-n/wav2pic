import os
import numpy as np
import random
from matplotlib.image import imread
from utils.loadder import *
from wav_process.wave_processing import *


class Get_Data():
    def __init__(self,path):
        self.filename = path
        self.data_num = 0
        self.dic_wav_list = {}
        self.dic_label_list = {}
        self.dic_wav_num = {}

    def get_num(self):
        self.dic_wav_list,self.dic_label_list,self.dic_wav_num = get_wav_list(self.filename)
        self.data_num = len(self.dic_wav_list)

    def read_data(self,n_start):
        wname = self.dic_wav_list[self.dic_wav_num[n_start]]
        lname = self.dic_label_list[self.dic_wav_num[n_start]]
        wavsignal,fs = read_wav_data(wname)
        data_input = GetFrequencyFeature(wavsignal, fs)
        data_input = data_input.T
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
        data_label = imread(lname)
        return data_input,data_label,wname


    def data_generator(self,batch_size=8):
        while True:
            inputs = np.zeros((batch_size, 441, 399, 1), dtype=float)
            outputs = np.zeros((batch_size, 144, 256, 3), dtype=int)
            for i in range(batch_size):
                rand_num = random.randint(0, self.data_num - 1)
                data_input, data_labels,wname = self.read_data(rand_num)
                try:
                    inputs[i, 0:len(data_input)] = data_input
                except:
                    print(wname)

                outputs[i, 0:len(data_labels)] = data_labels
            yield [inputs, outputs]
        pass

if (__name__=='__main__'):

    data = Get_Data(path='data/train.txt')
    data.get_num()
    inputs = data.data_generator(8)
    for x in inputs:
        y = x


    # for i in range(5):
    #     x = inputs.__next__()
    #     print(x)
    #     print('——————————————————————————————————————————————————————')