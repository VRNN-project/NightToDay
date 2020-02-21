from train_cycle_gan import *
from read_input import *
import numpy as np
import imageio
from skimage import img_as_ubyte

g_model_AtoB, g_model_BtoA, d_model_A, d_model_B = load_models(66, '/home/franco/models/2020-02-18-19:36:07')
evaluate_model(g_model_AtoB, datasets[0], test_fps_per_dataset[0][0:100])
