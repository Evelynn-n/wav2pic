from tensorflow.keras.layers import Layer,Conv2D,BatchNormalization,LeakyReLU,Add
from tensorflow.keras.layers import add
class BlockLayer(Layer):
    def __init__(self, d_model, stride,**kwargs):
        self.d_model = d_model
        self.stride = stride
        self.conv1 = Conv2D(4, kernel_size=1, strides=1)
        self.conv2 = Conv2D(4, kernel_size=3, strides=self.stride, padding="SAME")
        self.conv3 = Conv2D(d_model, kernel_size=1, strides=1)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.lkru1 = LeakyReLU()
        self.lkru2 = LeakyReLU()
        self.lkru3 = LeakyReLU()
        super(BlockLayer, self).__init__(**kwargs)
    def get_config(self):
        config = super(BlockLayer, self).get_config().copy()
        config.update({
            'd_model':self.d_model,
            'stride':self.stride
        })
        return config
    def call(self, inputs, *args, **kwargs):
        out = self.bn1(inputs)
        out = self.lkru1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lkru2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.lkru3(out)
        out = self.conv3(out)
        return out
    def compute_output_shape(self, input_shape):
        return input_shape
class ResBlock(Layer):
    def __init__(self,d_model,d_out,channels,**kwargs):
        self.d_model = d_model
        self.d_out = d_out
        self.channels = channels
        self.downsamping = Conv2D(d_out,1,strides=2,padding="SAME")
        self.upDim = Conv2D(d_model,1)
        self.bn = BatchNormalization()
        self.lkru = LeakyReLU()
        self.blockLayer = BlockLayer(self.d_out,self.d_out//self.d_model)
        super(ResBlock, self).__init__(**kwargs)
    def get_config(self):
        config = super(ResBlock, self).get_config().copy()
        config.update({
            'd_model':self.d_model,
            'd_out':self.d_out,
            'channels':self.channels
        })
        return dict(list(config.items()))
    def call(self, inputs, *args, **kwargs):
        if self.d_out//self.d_model != 1 :
            out = self.downsamping(inputs)
        elif inputs.shape[3]!=self.d_model:
            print(inputs.shape[3])
            inputs = self.upDim(inputs)
            out = inputs
        else:
            out = inputs
        temp =[self.blockLayer(inputs) for _ in range(self.channels)]
        result = add(temp)
        out =self.lkru(self.bn(out+result))
        return out
    def compute_output_shape(self, input_shape):
        return input_shape
def resnextLayer(inputs,kernel_size,out_size,channels):
    if inputs.shape[3]==kernel_size and out_size//kernel_size==1:
        temp = ResBlock(kernel_size,out_size,channels)(inputs)
        temp = ResBlock(kernel_size,out_size,channels)(temp)
        step= add([temp,inputs])
    elif out_size//kernel_size==1:
        temp = ResBlock(kernel_size, out_size, channels)(inputs)
        temp = ResBlock(out_size, out_size, channels)(temp)
        inputs = Conv2D(out_size, 1)(inputs)
        step = add([temp, inputs])
    else:
        temp = ResBlock(kernel_size,out_size,channels)(inputs)
        temp = ResBlock(out_size,out_size,channels)(temp)
        inputs = Conv2D(out_size,1,2)(inputs)
        step= add([temp,inputs])
    return step
def cbl(inputs,input_size,kernel_size,stride=1):
    convx = Conv2D(filters=input_size,kernel_size=kernel_size,strides=stride,padding="SAME")(inputs)
    bn = BatchNormalization()(convx)
    lkru = LeakyReLU()(bn)
    return lkru
def convBlock(inputs,input_size,output_size):
    if input_size==output_size:
        cbl1 = cbl(inputs,input_size/2,1)
        cbl2 = cbl(cbl1,input_size/2,3)
        cbl3 = cbl(cbl2,output_size,1)
        output = Add()([cbl3,inputs])
    else:
        cbl1 = cbl(inputs,input_size/2,1)
        cbl2 = cbl(cbl1,input_size/2,3,2)
        cbl3 = cbl(cbl2,output_size,1)
        conv = Conv2D(output_size,1,2)(inputs)
        output = Add()([cbl3,conv])
    return output
def resnext (inputs,kernel_size):
    x = resnextLayer(inputs,kernel_size,kernel_size,32)
    for i in range(3):
        kernel_size = kernel_size*2
        cbl1 = cbl(x, kernel_size / 8, 1)
        cbl2 = cbl(cbl1, kernel_size / 8, 3, 2)
        cbl3 = cbl(cbl2, kernel_size, 1)
        conv = Conv2D(kernel_size, 1, 2)(x)
        x = Add()([cbl3, conv])
        x = resnextLayer(x, kernel_size, kernel_size,32)
    return x
# def resnext (inputs,kernel_size,n_layers):
#     layer1,layer2,layer3,layer4 = n_layers
#     x = resnextLayer(inputs,kernel_size,kernel_size)
#     for i in range(layer1-1):
#         x = resnextLayer(x,kernel_size,kernel_size)
#     size =kernel_size*2
#     x = resnextLayer(x,kernel_size,size)
#     for _ in range(layer2-1):
#         x = resnextLayer(x,size,size)
#     x = resnextLayer(x,size,size*2)
#     size =size*2
#     for j in range(layer3-1):
#         x = resnextLayer(x,size,size)
#     x = resnextLayer(x,size,size*2)
#     size =size*2
#     for k in range(layer4-1):
#         x = resnextLayer(x,size,size)
#     return x