# Responsible for handling input dataset & data augmentation
import glob
import os
import numpy as np
import rawpy
from os import path
import matplotlib.pyplot as plt

# Short (input) and long (ground truth) exposure time photos directories
input_dir = '/home/franco/datasets/visualn/Sony/short/'
gt_dir = '/home/franco/datasets/visualn/Sony/long/'

train_fps = glob.glob(gt_dir + '*.ARW') # full paths
train_ids = [int(os.path.basename(train_fp)[0:5]) for train_fp in train_fps]

test_fps = glob.glob(input_dir + '*.ARW') # full paths
test_ids = [int(os.path.basename(test_fp)[0:5]) for test_fp in test_fps]

# print("train fps", train_fps)
# print(gt_dir)
# print("There are", len(test_ids), "test-ids and", len(train_ids), "train_ids\n\n\n\n")

patch_size = 128 # cropped image size


# pack Bayer image to 4 channels
def pack_raw_into4(raw):
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


def pack_raw_into3(raw):
    im = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=16)
    im = im.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)
    return im


# Reads input photo using a given path
def read_input(filepath, ratio=100, channels=4):
    raw_photo = rawpy.imread(filepath)
    if channels == 4:
        image = np.expand_dims(pack_raw_into4(raw_photo), axis=0) * ratio
    else:
        image = np.expand_dims(pack_raw_into3(raw_photo), axis=0) * ratio
    return image


# Reads ground truth photo using a given
# If half_size set to True, resulting photo resolution will be X x Y, not 2X x 2Y
def read_gt(filepath, half_size=False):
    gt_raw = rawpy.imread(filepath)
    im = gt_raw.postprocess(use_camera_wb=True, half_size=half_size, no_auto_bright=True, output_bps=16)
    return np.expand_dims(np.float32(im / 65535.0), axis=0)


# processes given image and its ground truth:
# - randomly crops 512x512 piece
# - random flip
# - random transpose
def augment_photos(input, gt):
    H = input.shape[1]
    W = input.shape[2]

    xx = np.random.randint(0, W - patch_size)
    yy = np.random.randint(0, H - patch_size)
    input_patch = input[:, yy:yy + patch_size, xx:xx + patch_size, :]
    gt_patch = gt[:, yy:yy + patch_size , xx:xx+patch_size, :]

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


# same method, but for a photo without a pair
def augment_photo(input):
    H = input.shape[1]
    W = input.shape[2]

    xx = np.random.randint(0, W - patch_size)
    yy = np.random.randint(0, H - patch_size)
    input_patch = input[:, yy:yy + patch_size, xx:xx + patch_size, :]

    if np.random.randint(2, size=1)[0] == 1:  # random flip
        input_patch = np.flip(input_patch, axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=2)
    if np.random.randint(2, size=1)[0] == 1:  # random transpose
        input_patch = np.transpose(input_patch, (0, 2, 1, 3))

    input_patch = np.minimum(input_patch, 1.0)
    return input_patch


# Generates samples. Used by generate_real_samples function in train_cycle_gan.
# If exposed set to True, these are ground truth (exposed) photos
class Generator:

    def __init__(self, fullpaths, gt_paths, gt_dir, exposed):
        self.fullpaths = fullpaths
        self.gt_paths = gt_paths
        self.exposed = exposed
        self.gt_dir = gt_dir

    def get_samples(self, n_samples):
        samples = []
        if not self.exposed:
            for _ in range(n_samples):
                sample_fp = np.random.choice(self.fullpaths)
                id = os.path.basename(sample_fp)[0:5]
                gt_files = glob.glob(gt_dir + '{}_00*.ARW'.format(id))
                gt_fp = gt_files[0]
                sample_exp = float(os.path.basename(sample_fp)[9:-5])
                gt_exp = float(os.path.basename(gt_fp)[9:-5])
                ratio = int(min(gt_exp / sample_exp, 300))
                sample_photo = read_input(sample_fp, ratio, 3)
                sample_photo = augment_photo(sample_photo)
                samples.append(sample_photo)
        else:
            for _ in range(n_samples):
                sample_fp = np.random.choice(self.gt_paths)
                sample_photo = read_gt(sample_fp, True)
                sample_photo = augment_photo(sample_photo)
                samples.append(sample_photo)
        return np.concatenate(samples)


def evaluate_photo(filepath, output_dir, model):
    id = path.basename(filepath)[0:5]
    gt_files = glob.glob(gt_dir + '{}_00*.ARW'.format(id))
    gt_fp = gt_files[0]
    sample_exp = float(path.basename(filepath)[9:-5])
    gt_exp = float(path.basename(gt_fp)[9:-5])
    ratio = min(gt_exp / sample_exp, 300)
    sample_photo = read_input(filepath, ratio, 3)
    gt_photo = read_gt(gt_fp, True)
    sample_photo, gt_photo = augment_photos(sample_photo, gt_photo)
    pred = model.predict(sample_photo)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(sample_photo[0, :, :, :])
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(gt_photo[0, :, :, :])
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(pred[0, :, :, :])
    plt.show()
