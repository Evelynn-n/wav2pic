import tensorflow as tf
# import cv2
import tensorflow.keras as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,LeakyReLU,BatchNormalization,Activation, MaxPooling2D, Dropout, Dense, Flatten, \
    GlobalAveragePooling2D,  SeparableConv2D, GlobalMaxPooling2D,Input,Layer,add
from tensorflow.keras import Model

def CNN(origin):
    inputs = Input(shape=(441,399,1))
    x = cbl_cell(inputs, 64, 3, 2)
    x = res_block(x, 32, 1)
    x = cbl_cell(x, 128, 1, 1)
    x = cbl_cell(x, 128, 3, 2)
    x = res_block(x, 64, 3)
    x = cbl_cell(x, 256, 1, 1)
    x = cbl_cell(x, 256, 3, 2)
    x = res_block(x, 128, 3)
    x = cbl_cell(x, 512, 1, 1)
    return  x

def cbl_cell(img_input,filters,kernels,stride=(1,1)):
    origin = Conv2D(filters, kernels,strides =stride,use_bias=False,padding='SAME')(img_input)
    x = BatchNormalization()(origin)
    x = LeakyReLU(alpha=0.05)(x)
    return x
def res(inputs,filter_num):
    x = cbl_cell(inputs,filter_num,1)
    x = cbl_cell(x,filter_num,3)
    x = add([inputs,x])
    return x
def res_block(inputs,filters,n):
    x = cbl_cell(inputs,filters,kernels=1)
    for i in range(n - 1):
        x = res(x, filters)
    x = Conv2D(filters,1,(1,1))(x)
    y = Conv2D(2*filters,1,(1,1))(inputs)
    x = K.backend.concatenate([x,y])
    return x
def RNN():
    pass
def transformer():
    pass
def VGGLoss():
    pass
def GAN(x):
    pass
cnn = CNN(0)
print(cnn.summary())