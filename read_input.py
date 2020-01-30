# Responsible for handling input dataset & data augmentation

import glob
import os
import numpy as np
import rawpy
import matplotlib.pyplot as plt

# Short (input) and long (ground truth) exposure time photos directories
input_dir = '/sata_disk/VRNN/Learning-to-See-in-the-Dark/dataset/Sony/short/'
gt_dir = '/sata_disk/VRNN/Learning-to-See-in-the-Dark/dataset/Sony/long/'

train_fps = glob.glob(gt_dir + '0*.ARW') # full paths
train_ids = [int(os.path.basename(train_fp)[0:5]) for train_fp in train_fps]

test_fps = glob.glob(input_dir + '1*.ARW') # full paths
test_ids = [int(os.path.basename(test_fp)[0:5]) for test_fp in test_fps]

patch_size = 512 # cropped image size


# pack Bayer image to 4 channels
def pack_raw(raw):
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


# Reads input photo using a given path
def read_input(filepath, ratio=100):
    raw_photo = rawpy.imread(filepath)
    image = np.expand_dims(pack_raw(raw_photo), axis=0) * ratio
    return image


# Reads ground truth photo using agiven path
def read_gt(filepath):
    gt_raw = rawpy.imread(filepath)
    im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    return np.expand_dims(np.float32(im / 65535.0), axis=0)


# processes given image and its ground truth:
# - randomly crops 512x512 piece
# - random flip
# - random transpose
def augment_photo(input, gt):
    H = input.shape[1]
    W = input.shape[2]

    xx = np.random.randint(0, W - patch_size)
    yy = np.random.randint(0, H - patch_size)
    input_patch = input[:, yy:yy + patch_size, xx:xx + patch_size, :]
    gt_patch = gt[:, yy * 2:yy * 2 + patch_size * 2, xx * 2:xx * 2 + patch_size * 2, :]

    if np.random.randint(2, size=1)[0] == 1:  # random flip
        input_patch = np.flip(input_patch, axis=1)
        gt_patch = np.flip(gt_patch, axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=2)
        gt_patch = np.flip(gt_patch, axis=2)
    if np.random.randint(2, size=1)[0] == 1:  # random transpose
        input_patch = np.transpose(input_patch, (0, 2, 1, 3))
        gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

    input_patch = np.minimum(input_patch, 1.0)
    return input_patch, gt_patch


# Auxiliary method - shows a plot comparing given image to its ground truth
def show_img(input, gt):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(input[0, :, :, :])
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(gt[0, :, :, :])
    plt.show()


# example
input_photo = read_input(os.path.join(input_dir, '00002_00_0.1s.ARW'))
gt_photo = read_gt(os.path.join(gt_dir, '00002_00_10s.ARW'))
input_photo, gt_photo = augment_photo(input_photo, gt_photo)
show_img(input_photo, gt_photo)