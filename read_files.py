import pandas as pd
import numpy as np
import pylab
from PIL import Image
from PIL import ImageOps
from scipy import misc

#mirror all the images so left and right opticus nerve is located at the same spot in every image

def read_training_set():
    labels = read_labels()
    #samples = read_sample_images()
    #samples = read_images()
    samples = read_images_3d()
    X = []
    y = []
    counter = range(2000)
    for ((name_left, img_left),(name_right, img_right)) in samples:
        row_left = labels[labels["image"] == name_left]
        label_left = row_left.values[0][1]
        X.append(img_left)
        vector = _label_to_vector(label_left)
        y.append(vector)

        row_right = labels[labels["image"] == name_right]
        label_right = row_right.values[0][1]
        X.append(img_right)
        vector = _label_to_vector(label_right)
        y.append(vector)

        counter.pop()
        if len(counter) == 0:
            counter = range(2000)
            yield _return_training_set(X,y)
            X = []
            y = []


    yield _return_training_set(X,y)
def _return_training_set(X,y):
    y = np.array(y)
    X = np.array(X)
    X = X.astype(np.float32)
    X = X.reshape(-1, 1, 100, 100)
    return X,y


def _label_to_vector(label):
    vector = np.array([0.0]*5)
    vector[label] = 1.0
    vector = vector.astype(np.float32)
    return vector

def read_labels():
    labels = pd.read_csv("trainLabels.csv")
    return labels

def read_images():
    #return _read_images(44350)
    return _read_images(44350, flatten=True)

def read_images_3d():
    #return _read_images(44350)
    return _read_images(1000, flatten=False)

def read_sample_images():
    return _read_images(20, flatten=True)

def read_sample_images_3d():
    return _read_images(20, flatten=False)

def _read_images(upper_limit, flatten=False):
    for index in xrange(10, upper_limit):
        try:
            left, right = _read_file_pair("train_set_resize", index, flatten)
        except IOError as e:
            continue
        yield left, right

def _read_file_pair(path, index, flatten):
    filename_right = "%s/%s_right.jpeg" % (path, index)
    filename_left = "%s/%s_left.jpeg" % (path, index)

    img_left = _read_image(filename_left, flatten=flatten)
    img_right = _read_image(filename_right, mirror=True, flatten=flatten)

    return (("%s_left" % index), img_left) , (("%s_right" % index), img_right)

def _read_image(filepath, flatten, mirror=False):
    #im = misc.imread(filepath, flatten=True)
    im = misc.imread(filepath)

    if flatten:
        im = im.flatten()

    im = im/255. #normalize
    im = im -im.mean()
    return im
