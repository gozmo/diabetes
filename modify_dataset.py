from conv_net import utils

dataset = "test_set"
targetset = "test_set_resize"
resolution = (100, 100)

utils.resize_dataset(dataset, targetset, resolution)
