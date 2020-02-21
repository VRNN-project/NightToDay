from train_cycle_gan import *
import numpy as np
import imageio
from skimage import img_as_ubyte
from read_input import *

g_model_AtoB, g_model_BtoA, d_model_A, d_model_B = load_models(66, '/home/franco/models/2020-02-18-19:36:07')

filepaths = test_fps_per_dataset[0][0:100]
data = [load_image_with_gt(filepath, g_model_AtoB, datasets[0]) for filepath in filepaths]

i = 0
for input, pred, output in data:
  imageio.imsave('{}_input.png'.format(i), ((input[0] * 255).astype(np.uint8)))
  imageio.imsave('{}_pred.png'.format(i), ((pred[0] * 255).astype(np.uint8)))
  imageio.imsave('{}_output.png'.format(i), ((output[0] * 255).astype(np.uint8)))
  i += 1
