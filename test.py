# Purpose: testing if the GPUs are working with tf

from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime, timezone
import tensorflow as tf
from read_input import datasets, train_fps_per_dataset, train_ids_per_dataset, test_fps_per_dataset, test_ids_per_dataset, Generator, evaluate_photo
import numpy as np

assert(len(tf.config.experimental.list_physical_devices('GPU')) > 0)


start = datetime.now(tz=timezone.utc)


input_generator = Generator(test_fps_per_dataset, train_fps_per_dataset, datasets, False)
input_generator2 = Generator(test_fps_per_dataset, train_fps_per_dataset, datasets, True)


start1 = datetime.now(tz=timezone.utc)
image = input_generator.get_samples(1, 0)
print('input avg, max, min', np.average(image), np.max(image), np.min(image))
stop1 = datetime.now(tz=timezone.utc)

start2 = datetime.now(tz=timezone.utc)
image = input_generator2.get_samples(1, 0)
print('output avg, max, min', np.average(image), np.max(image), np.min(image))
stop2 = datetime.now(tz=timezone.utc)

print("Generator on in {}s".format((start1 - start).total_seconds()))

print("Image Generator1 used {}s for 1 images".format((stop1 - start1).total_seconds()))
print("Image Generator2 used {}s for 1 images".format((stop2 - start2).total_seconds()))
