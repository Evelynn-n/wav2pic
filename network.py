import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from einops import rearrange, einops
from keras.layers import Rescaling

from tensorflow.keras.layers import Conv2D, LeakyReLU, PReLU, BatchNormalization, Activation, Dense, Flatten, \
    Input, add, concatenate, UpSampling2D
from tensorflow.keras import Model, initializers
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine.base_layer import Layer


from vit import MultiHeadSelfAttention
from wav_process.wave_processing import read_wav_data, GetFrequencyFeature, GetMfccFeature


def separate_heads(self, x, batch_size):
    x = tf.reshape(
        x, (batch_size, -1, self.num_heads, self.projection_dim)
    )
    return tf.transpose(x, perm=[0, 2, 1, 3])

class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它
    def get_config(self):
        config = super(Self_Attention, self).get_config().copy()
        config.update({
            'output_dim':self.output_dim,
        })
        return config
    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape", WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (self.output_dim ** 0.5)

        QK = K.softmax(QK)

        print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

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
    x = Dense(576)(x)
    return x
def attentionLayer(inputs,d_model,head):
    assert d_model% head ==0
    projection_dim = d_model // head
    q = Dense(d_model)(inputs)
    k = Dense(d_model)(inputs)
    v = Dense(d_model)(inputs)
    qw = K.backend.reshape(q, (-1, q.shape[1], head, projection_dim))
    kw = K.backend.reshape(k, (-1, k.shape[1], head, projection_dim))
    vw = K.backend.reshape(v, (-1, v.shape[1], head, projection_dim))
    qw = K.backend.permute_dimensions(qw, (0, 2, 1, 3))
    kw = K.backend.permute_dimensions(kw, (0, 2, 1, 3))
    vw = K.backend.permute_dimensions(vw, (0, 2, 1, 3))
    score = tf.matmul(qw, kw, transpose_b=True)
    dim_key = tf.cast(tf.shape(kw)[-1], tf.float32)
    scaled_score = score / tf.math.sqrt(dim_key)
    weight = K.backend.softmax(scaled_score, axis=-1)
    output = tf.matmul(weight, vw)
    attention = K.backend.permute_dimensions(output, (0, 2, 1, 3))
    attention = tf.reshape(attention, (-1,attention.shape[1], d_model))
    output = Dense(d_model)(attention)
    return output
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
def get_gan_network2(shape, generator, optimizer, vgg_loss):
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan = Model(inputs=gan_input, outputs=[x])
    gan.compile(loss=[vgg_loss],
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
        self.w_init = tf.keras.initializers.random_normal(stddev=0.02)
        self.g_init = tf.keras.initializers.random_normal(1., 0.02)

    def GAN_G(self, inputs):
        inputs = Input(inputs)
        # cnn = FNN(inputs)
        # x = rearrange(inputs, 'b h w c -> b c h w')
        # x = rearrange(inputs, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1=16, p2=16)
        x = tf.reshape(inputs,(-1,750,256))
        # # rescale = Rescaling(1. / 255)
        # x = Dense(256)(x)

        # att = MultiHeadSelfAttention(64)(inputs)
        # att = mhatt(64,8)(inputs)
        att = attentionLayer(x,256,8)
        layer2 = Dense(256)(att)
        addN = K.layers.add([x,layer2])
        att2 = attentionLayer(addN,256,8)
        addN = K.layers.add([x, att2])
        reshape = tf.reshape(addN, (-1, 400, 480, 1))
        des2 = Dense(3)(reshape)
        reshape = K.backend.permute_dimensions(des2, (0, 3, 1, 2))
        reshape = tf.reshape(reshape, (-1, 750, 768))
        att3 = attentionLayer(reshape, 768, 128)
        addN = K.layers.add([reshape, att3])
        att4 = attentionLayer(addN, 768, 128)
        addN = K.layers.add([reshape, att4])
        reshape = tf.reshape(addN, (-1, 400, 480, 3))
        des3 = Dense(3)(reshape)
        resize = tf.keras.layers.Reshape((36, 64, 250))(des3)
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
class transformer():
    pass

if (__name__ == '__main__'):
    from matplotlib.image import imread

    images = imread('video_process/labels/750_.jpg')
    wave_data, fs = read_wav_data('wav_process/wav/750_.wav')
    # freimg = GetFrequencyFeature(wave_data, fs)
    # freimg = freimg.T
    # freimg = freimg.reshape(freimg.shape[0], freimg.shape[1], 1)
    freimg1 = GetMfccFeature(wave_data, fs)
    freimg2 = GetFrequencyFeature(wave_data, fs)
    freimg = np.append(freimg1, freimg2, axis=1)
    freimg = np.pad(freimg, ((0, 1), (0, 0)), 'mean')
    # inputs = einops.rearrange(freimg, '(h p1) (w p2) -> (h w) (p1 p2)', p1=16, p2=16)
    freimg = freimg.T
    x = GAN_Network()
    gen = x.GAN_G(freimg.shape)
    des = x.GAN_D(images.shape)
    print(gen.summary())
    loss = VGGLoss(images.shape)
    gen.compile(loss=loss.vgg_loss, optimizer='adam')
    gen.save('attmodel.h5')
