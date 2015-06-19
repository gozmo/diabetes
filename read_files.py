import pandas as pd
import numpy as np
import pylab
from PIL import Image
from PIL import ImageOps
from scipy import misc
from conv_net.dataset import BaseDataset

class Dataset(BaseDataset):
    def read_training_set(self):
        self._start_idx = 0
        self._end_idx = 29700
        return self._read_dataset("train_set_resize", "trainLabels.csv")

    def read_validation_set(self):
        self._start_idx = 29700
        self._end_idx = 46000
        return self._read_dataset("validation_set_resize", "validationLabels.csv")

    def _read_dataset(self, path, label_path):
        labels = self._read_labels(label_path)
        samples = self._read_images(path)
        X = []
        y = []
        for ((name_left, img_left),(name_right, img_right)) in samples:
            row_left = labels[labels["image"] == name_left]
            label_left = row_left.values[0][1]
            X.append(img_left)
            vector = self._label_to_vector(label_left)
            y.append(vector)

            row_right = labels[labels["image"] == name_right]
            label_right = row_right.values[0][1]
            X.append(img_right)
            vector = self._label_to_vector(label_right)
            y.append(vector)

        print "_read_dataset, length", len(X)
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

    def _read_images(self, path):
        counter = 0
        for index in xrange(self._start_idx, self._end_idx):
            counter += 1
            if counter == self._training_set_size:
                break
            try:
                left_name, left, right_name, right = self._read_file_pair(path, index)
            except IOError as e:
                continue
            yield (left_name, left), (right_name, right)
        print counter

    def _read_file_pair(self, path, index):
        filename_right = "%s/%s_right.jpeg" % (path, index)
        filename_left = "%s/%s_left.jpeg" % (path, index)

        img_left = self._read_image(filename_left)
        img_right = self._read_image(filename_right)

        return ("%s_left" % index), img_left , ("%s_right" % index), img_right

    def _read_image(self, filepath):
        image = misc.imread(filepath)

        grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
        # get row number
        for rownum in range(len(image)):
               for colnum in range(len(image[rownum])):
                         grey[rownum][colnum] = np.average(image[rownum][colnum])

        grey = grey/255. #normalize
        grey = grey -grey.mean()
        return grey

    def read_test_set(self, test_set_size):
        self._test_set_size = test_set_size
        test_set_generator = self._test_set_generator()
        test_set = []
        test_set_file_names = []
        for left, left_name, right, right_name in self._test_set_generator():
            test_set.append(left)
            test_set_file_names.append(left_name)
            test_set.append(right)
            test_set_file_names(right_name)

        test_set = self._reshape_input_set(test_set)
        return test_set, test_set_file_names

    def _test_set_generator(self):
        for index in xrange(self._test_set_size):
            try:
                left, left_name, right, right_name = self._read_file_pair("test_set", index)
            except IOError as e:
                continue
            yield left, left_name, right, right_name


