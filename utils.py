import math

def relu(x):
    return max(0, x)

def sigmoid(x):
    return (1 / (1 + math.exp(-x)))

def featured_maps_size(matrix_size, filter_size, padding, stride):

    return int(((matrix_size - filter_size + (2 * padding)) / stride) + 1)


def print_matrix(matrix):
    mat_len = len(matrix)

    for i in range(mat_len):
        for j in range(mat_len):
            print(f"{matrix[i][j]}", end =" ")
        print("")