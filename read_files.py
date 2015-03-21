import pandas as pd
import numpy as np
import pylab
from PIL import Image
from PIL import ImageOps

#mirror all the images so left and right opticus nerve is located at the same spot in every image

def read_training_set():
    labels = read_labels()
    samples = read_sample_images()
    #samples = read_images()
    X = []
    y = []
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

    y = np.array(y)
    X = np.array(X)
    X = X.astype(np.float32)
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
    return _read_images(1000)

def read_sample_images():
    return _read_images(20)

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
    im = Image.open(open(filepath))
    if mirror:
        im = ImageOps.mirror(im)

    im = im.resize((300, 200))
    if flatten:
        (r,g,b) = im.split() #separate the differetn chanels
        fr=np.array(r,dtype=np.float32).flatten()
        fg=np.array(g,dtype=np.float32).flatten()
        fb=np.array(b,dtype=np.float32).flatten()
        im = np.concatenate((fr,fg,fb),axis=0)#we want theanos reshape to be able to separate R G and B later
    else:
        im = np.ndarray(im)

    im = im/255. #normalize
    im = im -im.mean()
    return im
