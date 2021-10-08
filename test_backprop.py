import numpy as np
from PIL import Image
from convolution_layer import *
from sequential import *
from dense_layer import *
from constant import *
from utils import *

if __name__ == '__main__':
    n_data = 10

    x_train = extract_mnist_images("train-images.idx3-ubyte", n_data)
    x_train = convert_to_grayscale(x_train) / 255
    y_train = extract_mnist_labels("train-labels.idx1-ubyte", n_data)
    y_train = one_hot_encoder(y_train)

    print("Image")
    print(x_train.shape)

    print("Label")
    print(y_train.shape)
    print(y_train)

    model = Sequential()

    convo_layer = ConvolutionLayer((5, 4), RELU, MAX, 2)
    dense_layer1 = DenseLayer(10, SOFTMAX)

    model.add_layer(convo_layer)
    model.add_layer(dense_layer1)

    batch_size = 5
    rate = 0.01
    model.train_model(x_train, y_train, batch_size, rate)
