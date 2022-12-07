import numpy as np
import tensorflow as tf
import tensorflow.keras as K
# from einops import rearrange, einops
# from keras.layers import Rescaling

from tensorflow.keras.layers import Conv2D, LeakyReLU, PReLU, BatchNormalization, Add, Dense, MaxPool2D, \
    Input, add, concatenate, UpSampling2D,LayerNormalization
from tensorflow.keras import Model, initializers
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine.base_layer import Layer
from networks.patchs_transformer import Encoder
from networks.resNeXt import resnext, resnextLayer
from vit import VisionTransformer
from wav_process.wave_processing import read_wav_data, GetFrequencyFeature, GetMfccFeature

class VGGLoss(object):
    def __init__(self, image_shape):
        self.image_shape = image_shape
        self.vgg = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
        vgg19 = self.vgg
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False

        return K.backend.mean(K.backend.square(model(y_true) - model(y_pred)))


def discriminator_block(model, filters, kernel_size, strides):
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model


def get_optimizer():
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam


class GAN_Network():
    def __init__(self):
        self.w_init = tf.keras.initializers.random_normal(stddev=0.02)
        self.g_init = tf.keras.initializers.random_normal(1., 0.02)

    def GAN_G(self, inputs):
        input = Input(inputs)
        kernel_size = 24
        # inputs = tf.reshape(inputs,(-1,400,480,1))
        inputs = K.layers.Reshape((400, 480, 1))(input)
        demo = Conv2D(kernel_size, 7, 2, padding="SAME")(inputs)
        demo = MaxPool2D(strides=2)(demo)
        x = resnextLayer(demo,kernel_size,kernel_size*2,32)
        x = resnextLayer(x, kernel_size*2, kernel_size * 4, 32)
        x = resnextLayer(x, kernel_size * 4, kernel_size * 8, 32)
        enc = Encoder(kernel_size * 8, 8, 6)(x)
        addL = Add()([enc, x])
        att_output = Conv2D(kernel_size * 16, 1)(addL)
        # att = MultiHeadAttention(8,384)(att_input)
        att_output = Conv2D(kernel_size * 32, 1)(att_output)
        resize = tf.keras.layers.Reshape((36, 64, 65))(att_output)
        # enc = Encoder(x,256,8)
        # dec = Decoder(enc,256,8)
        # enc = Encoder(dec,256,16)
        # dec = Decoder(enc,256,16)
        # addN = K.layers.add([x, dec])
        # attN = Transformer(750,8,256,6)(x)
        # reshape = K.layers.Reshape((750, 256, 1))(attN)
        # conv1 = Conv2D(64, (1, 1), (1, 1))(x)
        # conv2 = Conv2D(3, (1, 1), (1, 1))(conv1)
        # reshape = K.layers.Reshape((400, 480, 3))(conv2)
        # conv3 = Conv2D(3, (1, 1), (1, 1))(reshape)
        # resize = tf.keras.layers.Reshape((36, 64, 250))(conv3)
        n = Conv2D(64, 3, 1, padding='SAME')(resize)
        n = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(n)
        n = BatchNormalization()(n)
        temp = n
        for i in range(16):
            nn = Conv2D(64, (3, 3), (1, 1), padding='SAME')(n)
            nn = BatchNormalization()(nn)
            nn = Conv2D(64, (3, 3), (1, 1), padding='SAME')(nn)
            nn = BatchNormalization()(nn)
            nn = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(nn)
            nn = Add()([n, nn])
            n = nn
        nn = add([n, temp])
        n = Conv2D(256, (3, 3), (1, 1), padding='SAME')(nn)
        n = UpSampling2D(2)(n)
        n = Conv2D(256, (3, 3), (1, 1), padding='SAME')(n)
        n = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(n)
        n = UpSampling2D(2)(n)
        n = Conv2D(3, (1, 1), (1, 1), padding='SAME')(n)
        nn = LeakyReLU(alpha=0.05)(n)
        M = Model(input, nn)
        return M
if (__name__ == '__main__'):
    from matplotlib.image import imread

    images = imread('video_process/labels/750_.jpg')
    wave_data, fs = read_wav_data('wav_process/wav/750_.wav')
    freimg1 = GetMfccFeature(wave_data, fs)
    freimg2 = GetFrequencyFeature(wave_data, fs)
    freimg = np.append(freimg1, freimg2, axis=1)
    freimg = np.pad(freimg, ((0, 1), (0, 0)), 'mean')
    # inputs = einops.rearrange(freimg, '(h p1) (w p2) -> (h w) (p1 p2)', p1=16, p2=16)
    freimg = freimg.T
    x = GAN_Network()
    gen = x.GAN_G(freimg.shape)
    print(gen.summary())
    loss = VGGLoss(images.shape)
    gen.compile(loss=loss.vgg_loss, optimizer='adam')
    gen.save('resmodel.h5')
