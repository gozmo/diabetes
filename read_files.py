import pandas as pd
import numpy as np
import pylab
from PIL import Image
from PIL import ImageOps
from conv_net.dataset import BaseDataset
from os import listdir
from os.path import isfile, join

class Dataset(BaseDataset):
    def read_validation_set(self):
        return self._read_image_set("validation_set_resize", "validationLabels.csv")

    def read_training_set(self):
        return self._read_image_set("train_set_resize", "trainLabels.csv")

    def _read_image_set(self, directory_path, label_path):
        labels = self._read_labels(label_path)
        X = []
        y = []
        for name, level, index in zip(labels['image'], labels['level'], range(self._training_set_size)):
            target = self._label_to_vector(level)
            path = directory_path + "/" + name + ".jpeg"
            image = self._read_image(path)
            X.append(image)
            y.append(target)
        return self._return_training_set(X,y)

    def _return_training_set(self, X, y):
        y = np.array(y)
        X = self._reshape_input_set(X)
        return X,y

    def _reshape_input_set(self, X):
        X = np.array(X)
        X = X.astype(np.float32)
        if not self._flatten:
            X = X.reshape(-1, 1, self._height, self._width)
        return X

    def _label_to_vector(self, label):
        vector = np.array([0.0]*5)
        vector[label] = 1.0
        vector = vector.astype(np.float32)
        return vector

    def _read_labels(self, filename):
        labels = pd.read_csv(filename)
        return labels

    def read_test_set(self, directory, test_set_size=100000):
        files = self._list_files_in(directory)
        X = []
        image_names = []
        for image_name, index in zip(files, range(test_set_size)):
            image_path = directory + "/" + image_name
            image_name = image_name.replace(".jpeg","")
            image = self._read_image(image_path)
            X.append(image)
            image_names.append(image_name)
        return self._reshape_input_set(X), image_names

    def _list_files_in(self, directory):
        return [ f for f in listdir(directory) if isfile(join(directory,f)) ]

    def _test_set_generator(self):
        for index in xrange(self._test_set_size):
            try:
                left_name, left, right_name, right = self._read_file_pair("test_set_resize", index)
            except IOError as e:
                continue
            yield left_name, left, right_name, right
