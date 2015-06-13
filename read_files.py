import pandas as pd
import numpy as np
import pylab
from PIL import Image
from PIL import ImageOps
from scipy import misc
from conv_net.dataset import BaseDataset

class Dataset(BaseDataset):
    def read_training_set(self):
        labels = self._read_labels("trainLabels.csv")
        samples = self._read_images("train_set_resize")

        counter_function = lambda: range(20000)
        counter = counter_function()
        for ((name_left, img_left),(name_right, img_right)) in samples:
            vector = self.make_training_example(img_left, name_left, labels)
            vector = self.make_training_example(img_right, name_right, labels)

            counter.pop()
            if len(counter) == 0:
                counter = counter_function()
                yield self._return_training_set(X,y)
                self._X = []
                self._y = []
        yield self._return_training_set(X,y)

    def make_training_example(self, image, image_name, labels):
        training_row= labels[labels["image"] == image_name]
        label = training_row.values[0][1]
        vector = self._label_to_target_vector(label_left)
        self._X.append(image)
        self._y.append(vector)

    def _label_to_target_vector(self, label):
        vector = np.array([0.0]*5)
        vector[label] = 1.0
        vector = vector.astype(np.float32)
        return vector

    def _read_labels(self, filepath):
        labels = pd.read_csv(filepath)
        return labels

    def _read_images(self, path):
        for index in xrange(self._training_set_size):
            try:
                left, right = self._read_file_pair(path, index)
            except IOError as e:
                continue
            yield left, right

    def _read_file_pair(self, path, index):
        filename_right = "%s/%s_right.jpeg" % (path, index)
        filename_left = "%s/%s_left.jpeg" % (path, index)

        img_left = self._read_image(filename_left)
        img_right = self._read_image(filename_right)

        return (("%s_left" % index), img_left) , (("%s_right" % index), img_right)

    def read_validation_set(self):
        self._validation_set = self._read_images("validation_set")
        self._valudation_target = self._read_labels("validationLabels.csv")

    def get_validation_set(self):
        return self._validation_set
