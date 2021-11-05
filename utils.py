import math
import numpy as np
from PIL import Image
import csv


def relu(x):
    return max(0, x)


def dRelu(x: 'np.ndarray'):
    return np.where(x <= 0, 0, 1)


def dSigmoid(x: 'np.ndarray'):
    sigmoid_x = 1 / (1 + np.exp(-x))
    return sigmoid_x * (1 - sigmoid_x)


def sigmoid(x):
    return (1 / (1 + math.exp(-x)))


def tanh(x):
    return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))


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


def pad3D(inputs: 'np.ndarray', paddingSize: 'int') -> 'np.ndarray':
    paddedInputs = []
    for i in range(len(inputs)):
        paddedInputs.append(
            pad2D(inputs[i], paddingSize))
    return np.array(paddedInputs)


def pad2D(input: 'np.ndarray', paddingSize: 'int'):
    padding_dim = [(paddingSize, paddingSize),
                   (paddingSize, paddingSize)]
    return np.pad(input, padding_dim, mode='constant')


def extract_mnist_images(image_filepath, num_images=60000):
    _MNIST_IMAGE_SIZE = 28
    with open(image_filepath, "rb") as f:
        f.read(16)  # header
        buf = f.read(_MNIST_IMAGE_SIZE * _MNIST_IMAGE_SIZE * num_images)
        data = np.frombuffer(
            buf,
            dtype=np.uint8,
        ).reshape(num_images, _MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE)
        return data


def extract_mnist_labels(image_filepath, num_labels=60000):
    _MNIST_LABEL_SIZE = 1
    with open(image_filepath, "rb") as f:
        f.read(8)  # header
        buf = f.read(_MNIST_LABEL_SIZE * num_labels)
        data = np.frombuffer(
            buf,
            dtype=np.uint8,
        ).reshape(num_labels)
        return data


def one_hot_encoder(input: 'np.ndarray'):
    result = np.zeros((input.shape[0], 10), dtype=np.uint8)
    for i in range(input.shape[0]):
        result[i][input[i]] = 1
    return result


def cross_validation(k, dataset, seed=1, random=1):

    # Shuffle the dataset
    # np.random.seed(seed)
    # for i in range(random):
    #     np.random.shuffle(dataset)

    # Split into k groups
    k_group = np.array_split(dataset, k)

    test_data = k_group[0]
    train_data = k_group[1]
    for i in range(2, len(k_group)):
        train_data = np.concatenate((train_data, k_group[i]), axis=0)

    return test_data, train_data


def accuracy(test_values, predictions):
    N = test_values.shape[0]
    sum = 0
    for i in range(len(test_values)):
        if test_values[i] == np.argmax(predictions[i]):
            sum += 1
            
    accuracy = sum / N
    # accuracy = (test_values == predictions).sum() / N
    return accuracy


def convert_grayscale_to_rgb(image=None):
    stacked_img = np.stack((image,)*3, axis=1)
    return stacked_img


def convert_to_grayscale(image):
    return np.stack((image,)*1, axis=1)


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

    arr = extract_mnist_images("train-images.idx3-ubyte", 2)
    print(arr)
    print(arr.shape)

    stacked_arr = convert_grayscale_to_rgb(arr)
    print(stacked_arr.shape)
    print(stacked_arr[0].shape)

    # Test convert grayscale to RGB
    for i in stacked_arr:
        reshaped_i = np.reshape(i, (28, 28, 3))
        print(reshaped_i.shape)
        print(reshaped_i[0][0])
        img = Image.fromarray(reshaped_i, 'RGB')
        img.show()

def create_csv(predictions):
    fieldnames = ['id', 'labels']

    with open('predictions.csv', "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fieldnames)

        for i in range(len(predictions)):
            curr_row = [i+1, predictions[i]]
            csvwriter.writerow(curr_row)


def mmult(A, B):
    a_row = A.shape[0]
    a_col = A.shape[1]
    b_row = B.shape[0]
    b_col = B.shape[1]

    if (a_col != b_row):
        print("Cannot multiply. A_col != B_row")
        return

    # Result shape is a_row x b_col
    result = [[0 for i in range(b_col)] for j in range(a_row)]


    for row_idx_a in range(a_row):
        
        for col_idx_res in range(b_col):
            total = 0
            for col_idx_a in range(a_col):
                # Col idx a move along with row idx b
                total += A[row_idx_a][col_idx_a] * B[col_idx_a][col_idx_res]
            
            result[row_idx_a][col_idx_res] = total


    res_in_np_array = np.array(result)
    return res_in_np_array


def reverseMinMaxScaler(x, xMin, xMax):
    
    return (xMax-xMin) * x + xMin

def minMaxScaler(x, xMin, xMax):

    xVal = None

    if isinstance(x, str):
        xVal = float(x.replace(',', ''))
    else:
        xVal = x

    return (xVal - xMin) / (xMax - xMin)


def getMinMaxVal(array, rowLength, currCol):
    min = float('inf')
    max = float('-inf')

    value = None

    if isinstance(array[0][currCol], str):
        # new_array = []
        for i in range(rowLength):
            if array[i][currCol] == '-':
                continue
            value = float((array[i][currCol]).replace(',', ''))

            if value < min:
                min = value
            if value > max:
                max = value

    else:
        for i in range(rowLength):

            value = array[i][currCol]

            if value < min:
                min = value
            if value > max:
                max = value

    return min, max

