import tensorflow as tf
# import cv2
import tensorflow.keras as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,LeakyReLU,BatchNormalization,Activation, MaxPooling2D, Dropout, Dense, Flatten, \
    GlobalAveragePooling2D,  SeparableConv2D, GlobalMaxPooling2D,Input,Layer,add

def cbl_cell(img_input,filters,kernels,stride=(1,1)):
    origin = Conv2D(filters, kernels,strides =stride,use_bias=False,padding='SAME')(img_input)
    x = BatchNormalization()(origin)
    x = LeakyReLU(alpha=0.05)(x)
    return x
def res1(inputs,filter_num,stride=(1,1)):
    x = cbl_cell(inputs,filter_num,stride)
    x = cbl_cell(inputs,2*filter_num,stride)
    x = add([inputs,x])
    return x
def static_res_block(inputs,filters,n):
    x = res1(inputs, filters)
    for i in range(n - 1):
        x = res1(x, filters)
    return x
img_input = Input(shape=(441,399,1))
x = cbl_cell(img_input,32,(3,3))
x = cbl_cell(x,64,(3,3),(2,2))
x = static_res_block(x,32,1)
x = cbl_cell(x,128,(3,3),(2,2))
x = static_res_block(x,64,2)
x = cbl_cell(x,256,(3,3),(2,2))
x = static_res_block(x,128,8)
x = cbl_cell(x,512,(3,3),(2,2))
x = static_res_block(x,256,8)
x = cbl_cell(x,1024,(3,3),(2,2))
x = static_res_block(x,512,4)