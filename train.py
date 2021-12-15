from tensorflow.keras import Input
from matplotlib.image import imread
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
from network import VGGLoss, GAN_Network
from readdata import Get_Data
from utils.common import plot_generate_image
image_shape = (144,256,3)
input_shape = (441,399,1)
adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)
    gan.run_eagerly = True

    return gan

def train(batch_size,epochs):
    data = Get_Data(path='data/train.txt')
    data.get_num()
    length = data.data_num
    generator = data.data_generator(batch_size)
    loss = VGGLoss(image_shape)
    model = GAN_Network()
    gen = model.GAN_G(input_shape)
    dis = model.GAN_D(image_shape)
    gen.compile(loss=loss.vgg_loss, optimizer=adam)
    dis.compile(loss="binary_crossentropy", optimizer=adam)
    gan = get_gan_network(discriminator=dis, generator=gen, shape=input_shape,optimizer=adam, vgg_loss=loss.vgg_loss)

    for i in range(1,epochs+1):
        print('-' * 15, 'Epoch %d' % i, '-' * 15)
        for _ in tqdm(range(length//batch_size)):
            feature = generator.__next__()
            gen_x = feature[0]
            gen_y = feature[1]
            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            fake_data_Y = np.random.random_sample(batch_size) * 0.2
            gen_pred = gen.predict(gen_x)
            dis.trainable = True
            d_loss_real = dis.train_on_batch(gen_y,real_data_Y)
            d_loss_fake = dis.train_on_batch(gen_pred, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            dis.trainable=False
            gan_loss = gan.train_on_batch(gen_x, [gen_y, gan_Y])

        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        plot_generate_image(gen,gen_x[0],gen_y[0],i)
        loss_file = open('losses.txt', 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' % (i, gan_loss, discriminator_loss))
        loss_file.close()
        generator.save('gen_model%d.h5' % i)
        dis.save('dis_model%d.h5' % i)

        # if i % 5 == 0:
        #     generator.save( 'gen_model%d.h5' %i)
        #     dis.save('dis_model%d.h5' % i)






if(__name__=='__main__'):
    y = train(batch_size=8,epochs=1)