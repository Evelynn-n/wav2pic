import wave
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from scipy.fftpack import fft


def read_wav_data(filename):
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    '''
    wav = wave.open(filename, "rb")  # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes()  # 获取帧数
    print(num_frame)
    num_channel = wav.getnchannels()  # 获取声道数
    framerate = wav.getframerate()  # 获取帧速率
    print(framerate)
    num_sample_width = wav.getsampwidth()  # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame)  # 读取全部的帧
    wav.close()  # 关闭流
    wave_data = np.fromstring(str_data, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T  # 将矩阵转置
    # wave_data = wave_data
    return wave_data, framerate


def GetMfccFeature(wavsignal, fs):
    # 获取输入特征
    feat_mfcc = mfcc(wavsignal[0], fs)
    feat_mfcc_d = delta(feat_mfcc, 2)
    feat_mfcc_dd = delta(feat_mfcc_d, 2)
    # 返回值分别是mfcc特征向量的矩阵及其一阶差分和二阶差分矩阵
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature


def wav_show(wave_data, fs):  # 显示出来声音波形
    time = np.arange(0, len(wave_data)) * (1.0 / fs)  # 计算声音的播放时间，单位为秒
    # 画声音波形
    # plt.subplot(211)
    plt.plot(time, wave_data)
    # plt.subplot(212)
    # plt.plot(time, wave_data[1], c = "g")
    plt.show()


def wav_scale(energy):
    '''
    语音信号能量归一化
    '''
    for i in range(len(energy)):
        # if i == 1:
        #	#print('wavsignal[0]:\n {:.4f}'.format(energy[1]),energy[1] is int)
        energy[i] = float(energy[i]) / 100.0
    # if i == 1:
    #	#print('wavsignal[0]:\n {:.4f}'.format(energy[1]),energy[1] is int)
    return energy


x = np.linspace(0, 882 - 1, 882, dtype=np.int64)
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (882 - 1))  # 汉明窗


def GetFrequencyFeature(wavsignal, fs):
    if (44100 != fs):
        raise ValueError(
            '[Error] wav2pic currently only supports wav audio files with a sampling rate of 44100 Hz, but this audio is ' + str(
                fs) + ' Hz. ')

    # wav波形 加时间窗以及时移10ms
    time_window = 20  # 单位ms
    window_length = fs / 1000 * time_window  # 计算窗长度的公式，目前全部为882固定值

    wav_arr = np.array(wavsignal)
    # wav_length = len(wavsignal[0])
    wav_length = wav_arr.shape[1]

    range0_end = int(len(wavsignal[0]) / fs * 1000 - time_window) // 10 + 1  # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, int(window_length // 2)), dtype=np.float)  # 用于存放最终的频率特征数据
    data_line = np.zeros((1, int(window_length)), dtype=np.float)

    for i in range(0, range0_end):
        p_start = i * 441
        p_end = p_start + 882

        data_line = wav_arr[0, p_start:p_end]

        data_line = data_line * w  # 加窗

        data_line = np.abs(fft(data_line)) / wav_length

        data_input[i] = data_line[0: int(window_length // 2)]  # 设置为882除以2的值（即441）是取一半数据，因为是对称的
        # print(data_input.shape)
    data_input = np.log(data_input + 1)
    return data_input


if (__name__ == '__main__'):
    wave_data, fs = read_wav_data('125_.wav')
    wav_show(wave_data[0], fs)
    t0 = time.time()
    # freimg = GetMfccFeature(wave_data, fs)
    freimg = GetFrequencyFeature(wave_data, fs)
    t1 = time.time()
    print('time cost:', t1 - t0)
    freimg = freimg.T
    print(freimg.shape)
    plt.subplot(111)
    plt.imshow(freimg)
    plt.colorbar(cax=None, ax=None, shrink=0.5)
    plt.show()
