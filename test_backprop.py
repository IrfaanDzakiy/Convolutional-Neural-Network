import numpy as np
from PIL import Image
from convolution_layer import ConvolutionLayer
from constant import *
from utils import *

if __name__ == '__main__':
    n_data = 10

    x_train = extract_mnist_images("train-images.idx3-ubyte", n_data)
    x_train = convert_grayscale_to_rgb(x_train) / 255
    y_train = extract_mnist_labels("train-labels.idx1-ubyte", n_data)
    y_train = one_hot_encoder(y_train)

    print("Image")
    print(x_train.shape)

    print("Label")
    print(y_train.shape)
    print(y_train)
