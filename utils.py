import math
import numpy as np
from PIL import Image


def relu(x):
    return max(0, x)


def dRelu(x: 'np.ndarray'):
    return np.where(x <= 0, 0, 1)


def dSigmoid(x: 'np.ndarray'):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


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


def extract_mnist_images(image_filepath, num_images):
    _MNIST_IMAGE_SIZE = 28
    with open(image_filepath, "rb") as f:
        f.read(16)  # header
        buf = f.read(_MNIST_IMAGE_SIZE * _MNIST_IMAGE_SIZE * num_images)
        data = np.frombuffer(
            buf,
            dtype=np.uint8,
        ).reshape(num_images, _MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE)
        return data


def cross_validation(k, dataset, seed=1, random=1):

    # Shuffle the dataset
    np.random.seed(seed)
    for i in range(random):
        np.random.shuffle(dataset)

    # Split into k groups
    k_group = np.array_split(dataset, k)

    test_data = k_group[0]
    train_data = k_group[1]
    for i in range(2, len(k_group)):
        train_data = np.concatenate((train_data, k_group[i]), axis=0)

    return test_data, train_data


def accuracy(test_values, predictions):
    N = test_values.shape[1]
    accuracy = (test_values == predictions).sum() / N
    return accuracy


def convert_grayscale_to_rgb(image=None):
    stacked_img = np.stack((image,)*3, axis=-1)
    return stacked_img


if __name__ == '__main__':
    # Test k-cross validation , remove this later
    dataset = np.arange(90).reshape((10, 3, 3))
    test, train = cross_validation(10, dataset, 2)
    print(test)
    print(train)

    # Test accuracy , remove this later
    true_values = np.array([[1, 0, 0, 1, 1, 1, 1, 1, 1, 0]])
    predictions = np.array([[1, 0, 0, 1, 1, 1, 1, 1, 0, 1]])
    print(accuracy(true_values, predictions))

    arr = extract_mnist_images("train-images-idx3-ubyte", 2)
    print(arr)
    print(arr.shape)

    stacked_arr = convert_grayscale_to_rgb(arr)
    print(stacked_arr.shape)
    print(stacked_arr[0].shape)

    # Test convert grayscale to RGB
    for i in stacked_arr:
        img = Image.fromarray(i, 'RGB')
        img.show()
