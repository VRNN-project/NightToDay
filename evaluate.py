from train_cycle_gan import *
import numpy as np
import imageio
from skimage import img_as_ubyte

gen = Generator(test_fps_per_dataset, train_fps_per_dataset, datasets, False)

sample = gen.get_samples(5)
imageio.imsave('input1.png', img_as_ubyte(sample[0]))
# imageio.imsave('input2.png', img_as_ubyte(sample[1]))
# imageio.imsave('input3.png', img_as_ubyte(sample[2]))
# imageio.imsave('input4.png', img_as_ubyte(sample[3]))
# imageio.imsave('input5.png', img_as_ubyte(sample[4]))
# exit(0)


#g_model_AtoB, g_model_BtoA, d_model_A, d_model_B = load_models(18, '/home/franco/models/2020-02-18-17:11:16')
g_model_AtoB, g_model_BtoA, d_model_A, d_model_B = load_models(17, '/home/franco/models/2020-02-18-19:36:07')



#load_models(0, '/home/franco/models/2020-02-16-22:50:51')
x = g_model_AtoB.predict(sample)
print(x)
print(x.shape)
print('output avg, max, min', np.average(sample), np.max(sample), np.min(sample))
print('predicted avg, max, min', np.average(x), np.max(x), np.min(x))

x[0] = np.maximum(x[0], 0)
x[0] = np.minimum(x[0], 1)
#imageio.imsave('output.png', img_as_ubyte(x[0]))

imageio.imsave('output.png', ((x[0] * 255).astype(np.uint8)))
#imageio.imsave('input.png', img_as_ubyte(sample[0]))
