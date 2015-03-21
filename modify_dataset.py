from conv_net import utils

dataset = "train_set"
targetset = "train_set_resize"
resolution = (150, 100)

utils.resize_dataset(dataset, targetset, resolution)
