import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Activation, Concatenate, Conv2DTranspose, Input
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from random import randint, random
from numpy import ones, zeros, asarray
from read_input import train_ids, train_fps, test_fps, gt_dir, Generator, evaluate_photo
import time
from os import makedirs, path
import glob
import matplotlib as plt
from datetime import datetime

model_save_path = 'models/repos/NightToDay/saved_models/1581095043.509077'
epoch = 44

g_model_AtoB = tf.keras.models.load_model(path.join(model_save_path, 'g_model_AtoB_{}'.format(epoch) + '.h5'))


photo_path = '/disk-old/Sony/short/00001_00_0.1s.ARW'
evaluate_photo(photo_path, 'generated_photo/' + str(time.time()) , g_model_AtoB)