import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Activation, Concatenate, Conv2DTranspose, Input, Lambda
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from random import randint, random
from numpy import ones, zeros, asarray
from read_input import datasets, train_fps_per_dataset, train_ids_per_dataset, test_fps_per_dataset, test_ids_per_dataset, Generator, evaluate_photo
import time
from os import makedirs, path
import glob
import matplotlib as plt
from datetime import datetime
import numpy as np

def show_stats(name, x):
  print(name, round(np.average(x), 2), round(np.max(x), 2), round(np.min(x), 2))

def summarize_and_plot_model(model, plot_file):
    # summarize the model
    model.summary()
    # plot the model
    plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True)

lr = 0.0004 # 0.0002 domyślnie, chyba 03 było
beta_1 = 0.5

# Patch gan discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image inputsony_in
    in_image = Input(shape=image_shape)
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    # define model
    model = Model(in_image, patch_out)
    # compile model
    model.compile(loss='mse', optimizer=Adam(lr=lr, beta_1=beta_1), loss_weights=[0.5])
    return model



# generator a resnet block
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g


# define the standalone generator model
def define_generator(image_shape, n_resnet=9):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # c7s1-64
    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d128
    g = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256
    g = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # R256
    for _ in range(n_resnet):
        g = resnet_block(256, g)
    # u128
    g = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # u64
    g = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # c7s1-3
    g = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(g)

#    out_image = g
#    out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=init)(g)


    g = InstanceNormalization(axis=-1)(g)


    out_image = Activation('tanh')(g)
    out_image = Lambda(lambda x: (x + 1) / 2)(out_image)
    

#    out_image = Activation('tanh')(g)
#    out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=init)(g)
#    out_image = Lambda(lambda x: min(max(1, x), 0))(out_image)

#    out_image = g
    # define model
    model = Model(in_image, out_image)
    return model


# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False
    # discriminator element
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    # identity element
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    # forward cycle
    output_f = g_model_2(gen1_out)
    # backward cycle
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    # define model graph
    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    # define optimization algorithm configuration
    opt = Adam(lr=lr, beta_1=beta_1)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model

# c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])


# select a batch of random samples, returns images and target
def generate_real_samples(generator, n_samples, patch_shape):
    X = generator.get_samples(n_samples)
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return X, y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
    # generate fake instance
    X = g_model.predict(dataset)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool) - 1)
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)


# train cyclegan model
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA,
          n_train_samples, input_generator, gt_generator, starting_epoch, n_epochs, n_batch, model_save_path):

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_dir = 'logs/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # determine the output square shape of the discriminator
    n_patch = d_model_A.output_shape[1]
    # prepare image pool for fakes
    poolA, poolB = list(), list()
    # calculate the number of batches per training epoch
    bat_per_epo = int(n_train_samples / n_batch)
    bat_per_epo = 1000 # TODO: 1000
    # manually enumerate epochs
    for epoch in range(starting_epoch, n_epochs):
        print("Starting epoch {}".format(epoch))
        for bat in range(bat_per_epo):
            # select a batch of real samples
            X_realA, y_realA = generate_real_samples(input_generator, n_batch, n_patch)
            X_realB, y_realB = generate_real_samples(gt_generator, n_batch, n_patch)
            # generate a batch of fake samples
            X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
            X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)


            show_stats('      \t\t\t\t\t\t       XrealA ', X_realA[0])
            show_stats('      \t\t\t\t\t\t       XrealB ', X_realB[0])
            show_stats('      \t\t\t\t\t\t       XfakeA ', X_fakeA[0])
            show_stats('      \t\t\t\t\t\t       XfakeB ', X_fakeB[0])

            # update fakes from pool
            X_fakeA = update_image_pool(poolA, X_fakeA)
            X_fakeB = update_image_pool(poolB, X_fakeB)
            # update generator B->A via adversarial and cycle loss
            g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
            # update discriminator for A -> [real/fake]
            dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
            dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
            # update generator A->B via adversarial and cycle loss
            g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
            # update discriminator for B -> [real/fake]
            dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
            dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
            # summarize performance
            print('E %d B %d > dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (
                epoch, bat + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))

            # A - noc
            # B - dzień

            #valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id 
            with train_summary_writer.as_default():
                tf.summary.scalar('dA_l_realA', dA_loss1, step=epoch*bat_per_epo + bat)
                tf.summary.scalar('dA_l_fakeA', dA_loss2, step=epoch*bat_per_epo + bat)
                tf.summary.scalar('dB_l_realB', dB_loss1, step=epoch*bat_per_epo + bat)
                tf.summary.scalar('dB_l_fakeB', dB_loss2, step=epoch*bat_per_epo + bat)
                tf.summary.scalar('g_l_cAtoB', g_loss1, step=epoch*bat_per_epo + bat)
                tf.summary.scalar('g_l_cBtoA', g_loss2, step=epoch*bat_per_epo + bat)

        print("Saving models.")
        g_model_AtoB.save(path.join(model_save_path, 'g_model_AtoB_{}'.format(epoch) + '.h5'))
        g_model_BtoA.save(path.join(model_save_path, 'g_model_BtoA_{}'.format(epoch) + '.h5'))
        d_model_A.save(path.join(model_save_path, 'd_model_A_{}'.format(epoch) + '.h5'))
        d_model_B.save(path.join(model_save_path, 'd_model_B_{}'.format(epoch) + '.h5'))


# loading models
def load_models(start_epoch, load_models_path):
#    global g_model_AtoB, g_model_BtoA, d_model_A, d_model_B

    epoch = start_epoch
    model_save_path = load_models_path
    print("Loading models from epoch", epoch)

    g_model_AtoB = tf.keras.models.load_model(path.join(model_save_path, 'g_model_AtoB_{}'.format(epoch) + '.h5'), compile=False)
    g_model_AtoB.compile(loss='mse', optimizer=Adam(lr=lr, beta_1=beta_1), loss_weights=[0.5])

    g_model_BtoA = tf.keras.models.load_model(path.join(model_save_path, 'g_model_BtoA_{}'.format(epoch) + '.h5'), compile=False)
    g_model_BtoA.compile(loss='mse', optimizer=Adam(lr=lr, beta_1=beta_1), loss_weights=[0.5])

    d_model_A = tf.keras.models.load_model(path.join(model_save_path, 'd_model_A_{}'.format(epoch) + '.h5'))
    d_model_B = tf.keras.models.load_model(path.join(model_save_path, 'd_model_B_{}'.format(epoch) + '.h5'))
    return g_model_AtoB, g_model_BtoA, d_model_A, d_model_B


#load_models_path = "/home/franco/repos/NightToDay/models/repos/NightToDay/saved_models/1581095043.509077/"
#start_epoch = 47
#load_models(start_epoch, load_models_path)


if __name__ == "__main__":
    print(__name__)
    # input shape
    image_shape = (128, 128, 3)
    # generator: A -> B
    g_model_AtoB = define_generator(image_shape, 9) # TODO!!
    # generator: B -> A
    g_model_BtoA = define_generator(image_shape, 9) # TODO
    # discriminator: A -> [real/fake]
    d_model_A = define_discriminator(image_shape)
    # discriminator: B -> [real/fake]
    d_model_B = define_discriminator(image_shape)

    # dataset generators
    start_epoch = 15
    g_model_AtoB, g_model_BtoA, d_model_A, d_model_B = load_models(14, '/home/franco/models/2020-02-18-19:36:07')
    # composite: A -> B -> [real/fake, A]
    c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
    # composite: B -> A -> [real/fake, B]
    c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

    input_generator = Generator(test_fps_per_dataset, train_fps_per_dataset, datasets, False)
    gt_generator = Generator(test_fps_per_dataset, train_fps_per_dataset, datasets, True)

    n_epochs, n_batch = 200, 1
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    timestamp = "2020-02-18-19:36:07"
    models_path = path.join('/home/franco/models', timestamp)
    print("Models are being saved to {} after each epoch.".format(models_path))
#    makedirs(models_path)

    # train models
    train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA,
         sum([len(ids) for ids in train_ids_per_dataset]), input_generator, gt_generator, start_epoch, n_epochs, n_batch,
         models_path)

    # --- photo evaluation
    # evaluation_path = '/sata_disk/VRNN/Learning-to-See-in-the-Dark/evaluated_samples'
    evaluate_photo('/home/franco/datasets/vrnn/Sony/short/00001_00_0.1s.ARW', '', g_model_AtoB)
