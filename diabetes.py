import read_files
from conv_net.networks.net2 import Network
import numpy as np


def calculate_kappa(results_scalar, target_scalar):
    O_matrix = _create_o_matrix(results_scalar, target_scalar)

def _create_o_matrix(results_scalar, target_scalar):
    o_matrix = np.zeros((5,5))
    for prediction, target in zip(results_scalar, target_scalar):
        o_matrix[prediction, target] += 1
    return o_matrix

def _create_w_matrix(o_matrix):
    w_matrix = np.zeros((5,5))
    for x in xrange(5):
        for y in xrange(5):
            w_matrix[x,y] = (x-y) **2 / (5-1)**2
    return w_matrix

def _create_e_matrix(o_matrix):
    e_matrix = np.zeros((5,5))
    matrix_sum = float(o_matrix.sum())
    for x in xrange(5):
        for y in xrange(5):
            colum_sum = o_matrix[x,:].sum()
            row_sum = o_matrix[:y,].sum()
            e_matrix[x,y] = colum_sum*row_sum / matrix_sum
    return e_matrix

if __name__ == "__main__":
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

    results = net1.predict(X)
    results_scalar = [ result.argmax() for result in results]
    target_scalar = [target.argmax() for target in y]
