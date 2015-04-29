from conv_net.networks.net4 import Network
from kappa import calculate_kappa
from read_files import Dataset

if __name__ == "__main__":
    dataset = Dataset(False, training_set_size=50)
    print "Creating network",
    net = Network()
    print "done"
    print "Training network",
    for X,y in dataset.read_training_set():
        net.train(X,y)
    print "done"

    results = net.predict(X)
    results_scalar = [result.argmax() for result in results]
    target_scalar = [target.argmax() for target in y]

    print "result", results_scalar
    print "target", target_scalar
    print "kappa", calculate_kappa(results_scalar, target_scalar)
