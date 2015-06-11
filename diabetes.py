from conv_net.networks.net3 import Network
from conv_net import deep
from conv_net.utils import quadratic_weighted_kappa
from read_files import Dataset
from conv_net.utils import log

if __name__ == "__main__":
    log("Reading dataset. ")
    dataset = Dataset(False, training_set_size=60000)
    log("Done")

    scores = {}
    for network in deep.train_network(dataset):
        results_scalar = []
        target_scalar = []
        for X,y in dataset.read_training_set():
            results = network.predict(X)
            results_scalar += [result.argmax() for result in results]
            target_scalar += [target.argmax() for target in y]
        kappa = quadratic_weighted_kappa(results_scalar, target_scalar)
        scores[network.name] = kappa
        log("\tkappa: %s" % (kappa))

    log(scores)

