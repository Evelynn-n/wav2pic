import tensorflow.keras as K
from tensorflow.keras.layers import Conv2D, LeakyReLU, PReLU, BatchNormalization, Activation, Dense, Flatten, \
    Input, add, concatenate, UpSampling2D
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
from wav_process.wave_processing import read_wav_data, GetFrequencyFeature


def FNN(origin):
    # inputs = Input(shape=(441,399,1))
    x = cbl_cell(origin, 64, 3, 2)
    x = res_block(x, 32, 1)
    x = cbl_cell(x, 128, 1, 1)
    x = cbl_cell(x, 128, 3, 2)
    x = res_block(x, 64, 3)
    x = cbl_cell(x, 256, 1, 1)
    x = cbl_cell(x, 256, 3, 2)
    x = res_block(x, 128, 3)
    x = cbl_cell(x, 512, 1, 1)
    x = Dense(576, activation='relu')(x)
    return x

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

def cbl_cell(img_input, filters, kernels, stride=(1, 1)):
    origin = Conv2D(filters, kernels, strides=stride, use_bias=False, padding='SAME')(img_input)
    x = BatchNormalization()(origin)
    x = LeakyReLU(alpha=0.05)(x)
    return x


def res(inputs, filter_num):
    x = cbl_cell(inputs, filter_num, 1)
    x = cbl_cell(x, filter_num, 3)
    x = add([inputs, x])
    return x


def res_block(inputs, filters, n):
    x = cbl_cell(inputs, filters, kernels=1)
    for i in range(n - 1):
        x = res(x, filters)
    x = Conv2D(filters, 1, (1, 1))(x)
    y = Conv2D(2 * filters, 1, (1, 1))(inputs)
    x = concatenate([x, y])
    return x


class VGGLoss(object):
    def __init__(self, image_shape):
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
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
        self.w_init = K.initializers.random_normal(stddev=0.02)
        self.g_init = K.initializers.random_normal(1., 0.02)

    def GAN_G(self, inputs):
        inputs = Input(inputs)
        cnn = FNN(inputs)
        resize = K.layers.Reshape((36, 64, 700))(cnn)
        n = Conv2D(64, 3, 1, padding='SAME')(resize)
        n = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(n)
        temp = n
        for i in range(16):
            nn = Conv2D(64, (3, 3), (1, 1), padding='SAME')(n)
            nn = BatchNormalization()(nn)
            nn = Conv2D(64, (3, 3), (1, 1), padding='SAME')(nn)
            nn = BatchNormalization()(nn)
            nn = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(nn)
            nn = add([n, nn])
            n = nn
        n = Conv2D(64, (3, 3), (1, 1), padding='SAME')(n)
        n = BatchNormalization()(n)
        nn = add([n, temp])
        n = Conv2D(256, (3, 3), (1, 1), padding='SAME')(nn)
        n = UpSampling2D(2)(n)
        n = Conv2D(256, (3, 3), (1, 1), padding='SAME')(n)
        n = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(n)
        n = UpSampling2D(2)(n)
        n = Conv2D(3, (1, 1), (1, 1), padding='SAME')(n)
        nn = LeakyReLU(alpha=0.05)(n)
        M = Model(inputs, nn)
        return M

    def GAN_D(self, images):
        dis_input = Input(shape=images)

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(dis_input)
        model = LeakyReLU(alpha=0.2)(model)

        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)

        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        discriminator_model = Model(inputs=dis_input, outputs=model)

        return discriminator_model


if (__name__ == '__main__'):
    from matplotlib.image import imread

    images = imread('video_process/labels/750_.jpg')
    wave_data, fs = read_wav_data('wav_process/wav/750_.wav')
    freimg = GetFrequencyFeature(wave_data, fs)
    freimg = freimg.T
    freimg = freimg.reshape(freimg.shape[0], freimg.shape[1], 1)
    x = GAN_Network()
    gen = x.GAN_G(freimg)
    des = x.GAN_D(images)
    print(gen.summary())
