import pandas as pd
import numpy as np
import pylab
from PIL import Image
from PIL import ImageOps
from scipy import misc

class Dataset:
    def __init__(self, flatten, training_set_size=100, height=100, width=100):
        self._training_set_size = training_set_size
        self._height = height
        self._width = width
        self._flatten = flatten

    def read_training_set(self):
        self._labels = self._read_labels()
        samples = self._read_images()
        X = []
        y = []
        counter_function = lambda: range(4000)
        counter = counter_function()
        for ((name_left, img_left),(name_right, img_right)) in samples:
            row_left = self._labels[self._labels["image"] == name_left]
            label_left = row_left.values[0][1]
            X.append(img_left)
            vector = self._label_to_vector(label_left)
            y.append(vector)

            row_right = self._labels[self._labels["image"] == name_right]
            label_right = row_right.values[0][1]
            X.append(img_right)
            vector = self._label_to_vector(label_right)
            y.append(vector)

            counter.pop()
            if len(counter) == 0:
                counter = counter_function()
                yield self_return_training_set(X,y)
                X = []
                y = []
        yield self._return_training_set(X,y)

    def _return_training_set(self, X, y):
        y = np.array(y)
        X = np.array(X)
        X = X.astype(np.float32)
        print "before flatten", X.shape
        if not self._flatten:
            X = X.reshape(-1, 1, self._height, self._width)
        print "after flatten", X.shape
        return X,y

    def _label_to_vector(self, label):
        vector = np.array([0.0]*5)
        vector[label] = 1.0
        vector = vector.astype(np.float32)
        return vector

    def _read_labels(self):
        labels = pd.read_csv("trainLabels.csv")
        return labels

    def _read_images(self):
        for index in xrange(10, self._training_set_size):
            try:
                left, right = self._read_file_pair("train_set_resize", index)
            except IOError as e:
                continue
            yield left, right

    def _read_file_pair(self, path, index):
        filename_right = "%s/%s_right.jpeg" % (path, index)
        filename_left = "%s/%s_left.jpeg" % (path, index)

        img_left = self._read_image(filename_left)
        img_right = self._read_image(filename_right)

        return (("%s_left" % index), img_left) , (("%s_right" % index), img_right)

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
