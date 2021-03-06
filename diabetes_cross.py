from conv_net.networks.net3 import Network as Net3
from conv_net import deep
from read_files import Dataset
from conv_net.utils import log
from conv_net.crossval import crossvalidation

#TODO:
#-Validation set
#-pickle nets
#-ensemble
#-gradient boosting

if __name__ == "__main__":
    log("Reading dataset. ")
    size = 300
    log("size: %s" % size )
    dataset = Dataset(False, training_set_size=size)
    log("Done")

    average, scores = crossvalidation(dataset, Net3, 3)

    log(scores)
