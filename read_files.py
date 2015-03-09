import pandas as pd
import numpy as np
import pylab
from PIL import Image

#mirror all the images so left and right opticus nerve is located at the same spot in every image
HEIGHT=3168
WIDTH=4752

def read_training_set():
    labels = read_labels()
    samples = read_sample_images()
    X = []
    y = np.array([])
    for ((name_left, img_left),(name_right, img_right)) in samples:
        row_left = labels[labels["image"] == name_left]
        label_left = row_left.values[0][1]
        X.append(img_left)
        y = np.append(y, label_left)

        row_right = labels[labels["image"] == name_right]
        label_right = row_right.values[0][1]
        X.append(img_right)
        y = np.append(y, label_right)
    return X,y

def read_labels():
    labels = pd.read_csv("trainLabels.csv")
    return labels

def read_images():
    return _read_images(44350)

def read_sample_images():
    return _read_images(20)

def _read_images(upper_limit):
    for index in xrange(10, upper_limit):
        try:
            left, right = _read_file_pair("sample", index)
        except IOError as e:
            continue
        yield left, right

def _read_file_pair(path, index):
    filename_right = "%s/%s_right.jpeg" % (path, index)
    filename_left = "%s/%s_left.jpeg" % (path, index)

    img_left = _read_image(filename_left)
    img_right = _read_image(filename_right)

    return (("%s_left" % index), img_left) , (("%s_right" % index), img_right)

def _read_image(filepath):
    im = Image.open(open(filepath))
    (r,g,b) = im.split() #separate the differetn chanels
    fr=np.array(r,dtype=np.float32).flatten()
    fg=np.array(g,dtype=np.float32).flatten()
    fb=np.array(b,dtype=np.float32).flatten()
    feature = np.concatenate((fr,fg,fb),axis=0)#we want theanos reshape to be able to separate R G and B later
    feature = feature/255. #normalize
    feature = feature - feature.mean()
    return feature
