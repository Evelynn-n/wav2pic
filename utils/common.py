import matplotlib.pyplot as plt
import numpy as np
import os
def get_wav_list(filename):
    txt_obj = open(filename, 'r')  # 打开文件并读入
    txt_text = txt_obj.read()
    txt_lines = txt_text.split('\n')  # 文本分割
    dic_wavlist = {}  # 初始化字典
    dic_imglist = {}  # 初始化字典
    list_wavmark = []  # 初始化wav列表
    for i in txt_lines:
        if (i != ''):
            txt_l = i.split(' ')
            dic_wavlist[txt_l[0]] = txt_l[2]
            dic_imglist[txt_l[0]] = txt_l[1]
            list_wavmark.append(txt_l[0])
    txt_obj.close()
    return dic_wavlist,dic_imglist,list_wavmark
# y = get_wav_list('../data/train.txt')
path = 'outputs'
def plot_generate_image(generator,pred_data,origin_data,epoch):
    outputs = generator.predit(pred_data)
    plt.figure(figsize=(15,5))

    plt.subplot(1, 2, 1)
    plt.imshow(outputs[0], interpolation='nearest')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(origin_data, interpolation='nearest')
    plt.axis('off')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig( path+'/generated_image_%d.png' % epoch)
    plt.show()
