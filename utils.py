import math
import numpy as np


def relu(x):
    return max(0, x)


def sigmoid(x):
    return (1 / (1 + math.exp(-x)))


def softmax(input):
    result = []
    sum_of_exp_z = 0
    for i in range(len(input)):
        sum_of_exp_z += np.exp(input[i])

    for i in range(len(input)):
        result.append(np.exp(input[i]) / sum_of_exp_z)
    return result


def featured_maps_size(matrix_size, filter_size, padding, stride):

    return int(((matrix_size - filter_size + (2 * padding)) / stride) + 1)


def print_matrix(matrix):
    mat_len = len(matrix)

    for i in range(mat_len):
        for j in range(mat_len):
            print(f"{matrix[i][j]}", end=" ")
        print("")

def pad3D(inputs: 'np.ndarray', paddingSize: 'int'):
    paddedInputs = []
    for i in range(len(inputs)):
        paddedInputs.append(
            pad2D(inputs[i], paddingSize))
    return np.array(paddedInputs)


def pad2D(input: 'np.ndarray', paddingSize: 'int'):
    padding_dim = [(paddingSize, paddingSize),
                   (paddingSize, paddingSize)]
    return np.pad(input, padding_dim, mode='constant')

def _extract_mnist_images(image_filepath, num_images):
    _MNIST_IMAGE_SIZE = 28
    with open(image_filepath, "rb") as f:
        f.read(16)  # header
        buf = f.read(_MNIST_IMAGE_SIZE * _MNIST_IMAGE_SIZE * num_images)
        data = np.frombuffer(
            buf,
            dtype=np.uint8,
        ).reshape(num_images, _MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1)
        return data