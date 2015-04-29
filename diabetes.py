import read_files
from conv_net.networks.net4 import Network
from kappa import calculate_kappa

if __name__ == "__main__":
    print "Creating network",
    net = Network()
    print "done"
    print "Training network",
    for X,y in read_files.read_training_set(flatten=False, training_set_size=40):
        net.train(X,y)
    print "done"

    results = net.predict(X)
    results_scalar = [result.argmax() for result in results]
    target_scalar = [target.argmax() for target in y]

    print "result", results_scalar
    print "target", target_scalar
    print "kappa", calculate_kappa(results_scalar, target_scalar)
