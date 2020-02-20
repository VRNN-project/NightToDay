# Responsible for handling input dataset & data augmentation
import glob
import os
import numpy as np
import rawpy
from os import path
import matplotlib.pyplot as plt
from tensorflow import image


# Short (input) and long (ground truth) exposure time photos directories
sony_dataset = {
        'input_dir': '/home/franco/datasets/visualn/Sony/short/',
        'gt_dir': '/home/franco/datasets/visualn/Sony/long/',
        'extension': 'ARW',
        'name': 'Sony',
    }
# fuji_dataset = {
#         'input_dir': '/home/franco/datasets/visualn/Fuji/short/',
#         'gt_dir': '/home/franco/datasets/visualn/Fuji/long/',
#         'extension': 'RAF',
#         'name': 'Fuji',
#     }

datasets = [
    {
        'input_dir': '/home/franco/datasets/visualn/Sony/short/',
        'gt_dir': '/home/franco/datasets/visualn/Sony/long/',
        'extension': 'ARW',
        'name': 'Sony',
    },
    # {
    #     'input_dir': '/home/franco/datasets/visualn/Fuji/short/',
    #     'gt_dir': '/home/franco/datasets/visualn/Fuji/long/',
    #     'extension': 'RAF',
    #     'name': 'Fuji',
    # },
]

sony_gt_dir = '/home/franco/datasets/visualn/Sony/long/'
sony_input_dir = '/home/franco/datasets/visualn/Sony/short/'


train_fps_per_dataset = [glob.glob(dataset['gt_dir'] + '*.' + dataset['extension']) for dataset in datasets] # full paths
train_ids_per_dataset = [[int(os.path.basename(train_fp)[0:5]) for train_fp in train_fps] for train_fps in train_fps_per_dataset]

# TODO
test_fps_per_dataset = [glob.glob(dataset['input_dir'] + '*_0.1s.' + dataset['extension']) for dataset in datasets] # full paths
#test_fps_per_dataset = [glob.glob(dataset['input_dir'] + '*.' + dataset['extension']) for dataset in datasets] # full paths
test_ids_per_dataset = [[int(os.path.basename(test_fp)[0:5]) for test_fp in test_fps] for test_fps in test_fps_per_dataset]


patch_size = 128 # cropped image size


# pack Bayer image to 4 channels
def pack_raw_into4(raw):
    assert(False)
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
    assert(False)
    im = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=16)
    im = im.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)
    return im


# Reads input photo using a given path
def read_input(filepath, ratio=100, channels=4):
    print('read_input', filepath)
    raw_photo = rawpy.imread(filepath)
    if channels == 4:
        image = np.expand_dims(pack_raw_into4(raw_photo), axis=0) * ratio
    else:
        image = np.expand_dims(pack_raw_into3(raw_photo), axis=0) * ratio
    return image


# Reads ground truth photo using a given
# If half_size set to True, resulting photo resolution will be X x Y, not 2X x 2Y
def read_gt(filepath, half_size=False):
#    print('read_gt', filepath)
    gt_raw = rawpy.imread(filepath)                                             # CHANGED
    im = gt_raw.postprocess(use_camera_wb=True, half_size=half_size, no_auto_bright=False, output_bps=16)
    return np.expand_dims(np.float32(im / 65535.0), axis=0)


import cv2

# processes given image and its ground truth:
# - randomly crops 512x512 piece
# - random flip
# - random transpose

coeff = 16
crop = patch_size * 4

# same method, but for a photo without a pair
def augment_photo(input):
    H = input.shape[1]
    W = input.shape[2]

    xx = np.random.randint(0, W - crop)
    yy = np.random.randint(0, H - crop)

    input_patch = input[:, yy:yy + crop, xx:xx + crop, :]
    input_patch2 = []

    for i in range(len(input_patch)):
        input_patch2.append(cv2.resize(input_patch[i], 
                                    dsize=(patch_size, patch_size)))
    input_patch = np.array(input_patch2)

    if np.random.randint(2, size=1)[0] == 1:  # random flip
        input_patch = np.flip(input_patch, axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=2)
    if np.random.randint(2, size=1)[0] == 1:  # random transpose
        input_patch = np.transpose(input_patch, (0, 2, 1, 3))

    assert(np.max(input_patch) <= 1.000000001)
    input_patch = np.minimum(input_patch, 1.0)
    return input_patch

def augment_photos(input, gt):
    assert(False) # not to use
    H = input.shape[1]
    W = input.shape[2]

    xx = np.random.randint(0, W - crop)
    yy = np.random.randint(0, H - crop)

    input_patch = input[:, yy:yy + crop, xx:xx + crop, :]
    gt_patch = gt[:, yy:yy + crop , xx:xx + crop, :]


    input_patch = cv2.resize(input_patch, dsize=(patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    gt_patch = cv2.resize(gt_patch, dsize=(patch_size, patch_size), interpolation=cv2.INTER_CUBIC)

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

# Generates samples. Used by generate_real_samples function in train_cycle_gan.
# If exposed set to True, these are ground truth (exposed) photos
class Generator:

    def __init__(self, fullpaths_per_dataset, gt_paths_per_dataset, datasets, exposed):
        self.fullpaths_per_dataset = fullpaths_per_dataset
        self.gt_paths_per_dataset = gt_paths_per_dataset
        self.exposed = exposed
        self.datasets = datasets

    def get_samples(self, n_samples, dataset_no=None):
        if dataset_no is None:
            dataset_no = np.random.randint(0, len(self.datasets))
        dataset = self.datasets[dataset_no]
        fullpaths = self.fullpaths_per_dataset[dataset_no]
        gt_paths = self.gt_paths_per_dataset[dataset_no]
        samples = []
        if not self.exposed:
            for _ in range(n_samples):
                sample_fp = np.random.choice(fullpaths)
                # id = os.path.basename(sample_fp)[0:5]
                # gt_files = glob.glob(dataset['gt_dir'] + '{}_00*.{}'.format(id, dataset['extension']))
                # gt_fp = gt_files[0]
                # sample_exp = float(os.path.basename(sample_fp)[9:-5])
                # gt_exp = float(os.path.basename(gt_fp)[9:-5])
                # ratio = int(min(gt_exp / sample_exp, 300))
                # sample_photo = read_input(sample_fp, ratio, 4)
                sample_photo = read_gt(sample_fp, True) # DIFF!
                sample_photo = augment_photo(sample_photo)
                samples.append(sample_photo)
        else:
            for _ in range(n_samples):
                sample_fp = np.random.choice(gt_paths)
                sample_photo = read_gt(sample_fp, True)
                sample_photo = augment_photo(sample_photo)
                samples.append(sample_photo)
        return np.concatenate(samples)


def evaluate_photo(filepath, output_dir, model, dataset):
    id = path.basename(filepath)[0:5]
    gt_files = glob.glob(dataset['gt_dir'] + '{}_00*.{}'.format(id, dataset['extension']))
    gt_fp = gt_files[0]
    sample_exp = float(path.basename(filepath)[9:-5])
    gt_exp = float(path.basename(gt_fp)[9:-5])
    ratio = min(gt_exp / sample_exp, 300)
    sample_photo = read_input(filepath, ratio, 4)
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
    print("Going to show image")
    plt.show()
    print("Image shown")


print('dataset_size', sum([len(ids) for ids in train_ids_per_dataset]))


# Calculates mean psnr (https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) in a dataset
def mean_psnr(gt_photos, outputs):
    return np.mean(image.psnr(gt_photos, outputs, 1))


# Calculates mean ssim (https://en.wikipedia.org/wiki/Structural_similarity) in a dataset
def mean_ssim(gt_photos, outputs):
    return np.mean(image.ssim(gt_photos, outputs, 1))


# Calculates two measures above (psnr, ssim) for a given lists of paired photos (ground_truth, output)
def calculate_measures(dataset):
    outputs, ground_truths = ([ i for i, _ in dataset ], [ j for _, j in dataset ] )
    return mean_psnr(ground_truths, outputs), mean_ssim(ground_truths, outputs)


# Loads output photo with its ground_truth
def load_image_with_gt(filepath, model, dataset):
    id = path.basename(filepath)[0:5]
    gt_files = glob.glob(dataset['gt_dir'] + '{}_00*.{}'.format(id, dataset['extension']))
    gt_fp = gt_files[0]
    sample_exp = float(path.basename(filepath)[9:-5])
    gt_exp = float(path.basename(gt_fp)[9:-5])
    ratio = min(gt_exp / sample_exp, 300)
    sample_photo = read_input(filepath, ratio, 4)
    gt_photo = read_gt(gt_fp, True)
    sample_photo, gt_photo = augment_photos(sample_photo, gt_photo)
    pred = model.predict(sample_photo)
    return pred, gt_photo



# Evaluates test dataset given paths to test files
def evaluate_model(model, test_data, filepaths):
    loaded_photos = [load_image_with_gt(filepath, model, test_data) for filepath in filepaths]
    measures = calculate_measures(loaded_photos)
    print("PSNR: {}, SSIM: {}")
    return measures


# Tu sobie mozemy odkomentowac i policzyc
# evaluate_model(model, datasets[0], test_fps_per_dataset)
