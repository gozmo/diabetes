import read_files
from conv_net.networks.net1 import Network
import numpy as np

print "Loading training set",
X,y = read_files.read_training_set()
print "done"

input_size = len(X[0])
output_size = len(y[0])

print "Creating network",
net1 = Network()
print "done"
print "Training network",
net1.train(X,y)
print "done"
