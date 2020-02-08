import glob
import os
import numpy as np
import imageio
from skimage import img_as_ubyte
from read_input import gt_dir, test_fps, read_input, read_gt, augment_photos
import matplotlib.pyplot as plt


# generate samples the dataset
# channels - number of channels to pack short exposure photos into
# If half_size set to True, resulting ground truth photo resolution will be X x Y, not 2X x 2Y (comparing to input)
def generate_samples(n=200, samples_dir='~/datasets/vrnn/samples/',
                     channels=4, half_size=True):
    input_dir = os.path.join(samples_dir, 'short')
    sample_gt_dir = os.path.join(samples_dir, 'long')
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir)
        os.makedirs(os.path.join(input_dir, '100'))
        os.makedirs(os.path.join(input_dir, '250'))
        os.makedirs(os.path.join(input_dir, '300'))

    if not os.path.isdir(sample_gt_dir):
        os.makedirs(sample_gt_dir)
        os.makedirs(os.path.join(sample_gt_dir, '100'))
        os.makedirs(os.path.join(sample_gt_dir, '250'))
        os.makedirs(os.path.join(sample_gt_dir, '300'))

    for i in range(n):
        print("Generating sample nr {}".format(i))
        sample_fp = np.random.choice(test_fps)
        id = os.path.basename(sample_fp)[0:5]
        gt_files = glob.glob(gt_dir + '{}_00*.ARW'.format(id))
        gt_fp = gt_files[0]
        sample_exp = float(os.path.basename(sample_fp)[9:-5])
        gt_exp = float(os.path.basename(gt_fp)[9:-5])
        ratio = min(gt_exp / sample_exp, 300)
        sample_photo = read_input(sample_fp, ratio, channels)
        gt_photo = read_gt(gt_fp, half_size)
        sample_photo, gt_photo = augment_photos(sample_photo, gt_photo)
        sample_photo = sample_photo[0, :, :, :]
        gt_photo = gt_photo[0, :, :, :]
        curr_input_dir = os.path.join(input_dir, str(int(ratio)))
        curr_gt_dir = os.path.join(sample_gt_dir, str(int(ratio)))
        output_sample_path = os.path.join(curr_input_dir, os.path.basename(sample_fp))[:-4] + '_{}.png'.format(i)
        output_gt_path = os.path.join(curr_gt_dir, os.path.basename(gt_fp))[:-4] + '_{}.png'.format(i)
        imageio.imsave(output_sample_path, img_as_ubyte(sample_photo))
        imageio.imsave(output_gt_path, img_as_ubyte(gt_photo))


# Auxiliary method - shows a plot comparing given image to its ground truth
def show_img(input, gt):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(input[0, :, :, :])
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(gt[0, :, :, :])
    plt.show()


generate_samples(30, '-in-the-Dark/samples6/', 3)
