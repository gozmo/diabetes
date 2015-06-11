from conv_net.networks.net3 import Network as Network3
from conv_net import deep
from conv_net.utils import quadratic_weighted_kappa
from read_files import Dataset
from conv_net.utils import log
from crossval import crossvalidation

if __name__ == "__main__":
    log("Reading dataset. ")
    dataset = Dataset(False, training_set_size=600)
    log("Done")

    average, scores = crossvalidation(dataset, Network3, 3)

    log(scores)
