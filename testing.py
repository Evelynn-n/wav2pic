from tensorflow.keras import Input
from matplotlib.image import imread
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from network import GAN_Network,VGGLoss
from readdata import Get_Data
from wav_process.wave_processing import read_wav_data, GetFrequencyFeature
def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan
adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
images = imread('video_process/labels/750_.jpg')
wave_data, fs = read_wav_data('wav_process/wav/750_.wav')
data = Get_Data(path='data/train.txt')
data.get_num()
inputs = data.data_generator()
freimg = GetFrequencyFeature(wave_data, fs)
freimg = freimg.T
freimg = freimg.reshape(freimg.shape[0],freimg.shape[1],1)
inputs = Input(freimg.shape)
loss = VGGLoss(freimg.shape)
x = GAN_Network()
gen = x.GAN_G(inputs)
des = x.GAN_D(images)
gen.compile(loss = loss.vgg_loss,optimizer=adam)
des.compile(loss = "binary_crossentropy",optimizer=adam)
gan = get_gan_network(des, freimg.shape, gen, adam, loss.vgg_loss)
print(gen.summary())

#
# for _ in tqdm(range(batch_count)):
#     pass

import numpy as np
batch_size = 8
real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
fake_data_Y = np.random.random_sample(batch_size)*0.2