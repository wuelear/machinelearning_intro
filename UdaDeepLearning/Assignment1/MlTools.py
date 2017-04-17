
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


class MlTools:
    #class members considered private
    __url = 'http://commondatastorage.googleapis.com/books1000/'
    __last_percent_reported = None
    __data_root = '/home/learner/Udacity/MachineLearning/Data' # Change me to store data elsewhere
    __num_classes = 10
    __image_size = 28  # Pixel width and height.
    __pixel_depth = 255.0  # Number of levels per pixel.


    def __init__(self):
        np.random.seed(133)
    def set_data_root(self, dataRoot):
        self.__data_root = dataRoot
    def set_num_classes(self, numClasses):
        self.__num_classes = numClasses
    def set_image_size(self, imageSize):
        self.__image_size = imageSize
    def set_pixel_depth(self, pixelDepth):
        self.__pixel_depth = pixelDepth


    def download_progress_hook(self, count, blockSize, totalSize):
        """
            Purpose:
            A hook to report the progress of a download. This is mostly intended for users with
            slow internet connections. Reports every 5% change in download progress.
        """
        global last_percent_reported
        percent = int(count * blockSize * 100 / totalSize)

        if last_percent_reported != percent:
            if percent % 5 == 0:
                sys.stdout.write("%s%%" % percent)
                sys.stdout.flush()
            else:
                sys.stdout.write( "." )
                sys.stdout.flush()

            last_percent_reported = percent


    def maybe_download(self, filename, expected_bytes, force=False):
        """
            Purpose:
                - Download a file from __url if not present.
                - Make sure it's the right size.
        """
        dest_filename = os.path.join(self.__data_root, filename)

        if force or not os.path.exists(dest_filename):
            print('Attempting to download:', filename)
            filename, _ = urlretrieve(self.__url + filename, dest_filename, reporthook=self.download_progress_hook)
            print('\nDownload Complete!')

        statinfo = os.stat(dest_filename)

        if statinfo.st_size == expected_bytes:
            print('Found and verified', dest_filename)
        else:
            raise Exception(
                'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
        return dest_filename


    def maybe_extract(self, filename, force=False):
        """

        """
        root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz

        if os.path.isdir(root) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping extraction of %s.' % (root, filename))
        else:
            print('Extracting data for %s. This may take a while. Please wait.' % root)
            tar = tarfile.open(filename)
            sys.stdout.flush()
            tar.extractall(self.__data_root)
            tar.close()

        data_folders = [ os.path.join(root, d) for d in sorted(os.listdir(root))
                         if os.path.isdir(os.path.join(root, d)) ]

        if len(data_folders) != self.__num_classes:
            raise Exception(
                'Expected %d folders, one per class. Found %d instead.' % (self.__num_classes, len(data_folders)))
        print(data_folders)
        return data_folders


    def load_letter(self, folder, min_num_images):
        """Load the data for a single letter label."""
        image_files = os.listdir(folder)
        image_size = self.__image_size
        pixel_depth = self.__pixel_depth

        dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
        print(folder)
        num_images = 0

        for image in image_files:
            image_file = os.path.join(folder, image)
            try:
                image_data = ndimage.imread(image_file).astype(float)
                #In order to reduce mean and std deviation we:
                #- map the original value range 0-255 to -0.5 to 0.5
                image_data = (image_data - pixel_depth / 2) / pixel_depth
                if image_data.shape != (image_size, image_size):
                  raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[num_images, :, :] = image_data
                num_images = num_images + 1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

            dataset = dataset[0:num_images, :, :]
            if num_images < min_num_images:
              raise Exception('Many fewer images than expected: %d < %d' %
                              (num_images, min_num_images))

            print('Full dataset tensor:', dataset.shape)
            print('Mean:', np.mean(dataset))
            print('Standard deviation:', np.std(dataset))
            return dataset


    def maybe_pickle(self, data_folders, min_num_images_per_class, force=False):
        dataset_names = []
        for folder in data_folders:
            set_filename = folder + '.pickle'
            dataset_names.append(set_filename)
            if os.path.exists(set_filename) and not force:
                # You may override by setting force=True.
                print('%s already present - Skipping pickling.' % set_filename)
            else:
                print('Pickling %s.' % set_filename)
                dataset = self.load_letter(folder, min_num_images_per_class)
                try:
                    with open(set_filename, 'wb') as f:
                        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print('Unable to save data to', set_filename, ':', e)

        return dataset_names


