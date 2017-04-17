from __future__ import print_function
__author__ = 'xxxh'

#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage

import Assignment1.MlTools as tl


# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline

tool = tl.MlTools()

pixel_depth = 255
tool.set_data_root('/home/learner/Udacity/MachineLearning/Data')
tool.set_num_classes(10)
tool.set_image_size(28)
tool.set_pixel_depth(pixel_depth)


train_filename = tool.maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = tool.maybe_download('notMNIST_small.tar.gz', 8458043)

train_folders = tool.maybe_extract(train_filename)
test_folders = tool.maybe_extract(test_filename)

train_datasets = tool.maybe_pickle(train_folders, 45000)
test_datasets = tool.maybe_pickle(test_folders, 1800)

for folder in train_folders:
    image_files = os.listdir(folder)
    for image in image_files:
        image_file = os.path.join(folder, image)

        image_data = ndimage.imread(image_file).astype(float)
        print('Mean:', np.mean(image_data))
        print('Standard deviation:', np.std(image_data))

        image_data_normalized = (image_data - pixel_depth / 2) / pixel_depth
        print('Mean:', np.mean(image_data_normalized))
        print('Standard deviation:', np.std(image_data_normalized))

        plt.subplot(121)
        plt.imshow(image_data)
        plt.subplot(122)
        plt.imshow(image_data_normalized)
        plt.show()