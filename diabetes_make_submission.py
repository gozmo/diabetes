from conv_net.networks.net3 import Network as Net3
from conv_net.networks.net7 import Network as Net7
from conv_net import deep
from read_files import Dataset
from conv_net.utils import log
from conv_net.crossval import crossvalidation
import csv

if __name__ == "__main__":
    log("Reading dataset. ")
    size = 30000
    log("size: %s" % size )
    dataset = Dataset(False, training_set_size=size)

    network = Net7()
    X,y = dataset.read_training_set()
    log("Done")

    log("Training network")
    network.train(X,y)

    log("Predicting test set")

    filename = "submission.csv"
    with open(filename, 'w') as csvfile:
        fieldnames = ["image", "level"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        test_set, test_names = dataset.read_test_set("test_set_resize", 100)
        results = network.predict(test_set)
        results_scalar = [result.argmax() for result in results]

        for name, prediction in zip(test_names, results_scalar):
            writer.writerow({'image': name, 'level': prediction})
