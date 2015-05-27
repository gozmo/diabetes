from conv_net.networks.net3 import Network
from kappa import quadratic_weighted_kappa
from read_files import Dataset

if __name__ == "__main__":
    dataset = Dataset(False, training_set_size=200)
    net = Network()
    for X,y in dataset.read_training_set():
        net.train(X,y)

    results_scalar = []
    target_scalar = []
    for X,y in dataset.read_training_set():
        results = net.predict(X)
        results_scalar += [result.argmax() for result in results]
        target_scalar += [target.argmax() for target in y]

    print len(results_scalar)
    print len(target_scalar)
    print "kappa\n", quadratic_weighted_kappa(results_scalar, target_scalar)
