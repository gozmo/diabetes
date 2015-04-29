import numpy as np

def calculate_kappa(results_scalar, target_scalar):
    o_matrix = _create_o_matrix(results_scalar, target_scalar)
    e_matrix = _create_e_matrix(o_matrix)
    w_matrix = _create_w_matrix(o_matrix.shape)

    print "o_matrix", o_matrix
    print "e_matrix", e_matrix
    print "w_matrix", w_matrix

    return 1 - (_sum_matrix(w_matrix, o_matrix) / _sum_matrix(w_matrix, e_matrix))

def _create_o_matrix(results_scalar, target_scalar):
    size = 5
    o_matrix = np.zeros((size,size))
    for prediction, target in zip(results_scalar, target_scalar):
        o_matrix[prediction, target] += 1
    return o_matrix

def _create_e_matrix(o_matrix):
    e_matrix = np.zeros(o_matrix.shape)
    matrix_sum = float(o_matrix.sum())
    for x in xrange(o_matrix.shape[0]):
        for y in xrange(o_matrix.shape[1]):
            column_sum = float(o_matrix[x,:].sum())
            row_sum = float(o_matrix[:y,].sum())
            e_matrix[x][y] = column_sum*row_sum / matrix_sum
    return e_matrix

def _create_w_matrix(shape):
    w_matrix = np.zeros(shape)
    for x in xrange(shape[0]):
        for y in xrange(shape[1]):
            w_matrix[x][y] = (float(x)-y)**2 / (shape[0]-1)**2
    return w_matrix

def _sum_matrix(w_matrix, matrix):
    matrix_sum = 0.0
    for row_idx in range(len(w_matrix)):
        for column_idx in range(len(w_matrix[row_idx])):
            matrix_sum += w_matrix[row_idx][column_idx] * matrix[row_idx][column_idx]
    return matrix_sum
