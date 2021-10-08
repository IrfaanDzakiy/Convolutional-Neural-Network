from utils import *
from convolution_layer import *
from dense_layer import *
from sequential import *
from PIL import Image
import numpy as np

mat_example = [
    [4, 3, 8, 5],
    [9, 1, 3, 6],
    [6, 3, 5, 3],
    [2, 5, 2, 5]
]

inputs = np.random.randint(
    256, size=(1, 32, 32))


def main():
    inputs = extract_mnist_images("train-images-idx3-ubyte", 2)

    print(f"Total images : {len(inputs)}")
    
    for image in inputs:
        convoLayer = ConvolutionLayer((3, 2), RELU, MAX, 2)
        denseLayer = DenseLayer(120, "sigmoid")

        model = Sequential()
        model.add_inputs(image)
        model.add_layer(convoLayer)
        model.add_layer(denseLayer)
        model.calculate()
        model.print_summary()


def test_backprop_dense():

    data = [[np.random.rand() for j in range(10)] for i in range(20)]
    targets = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    model = Sequential()

    dense_layer1 = DenseLayer(10, SIGMOID)
    dense_layer2 = DenseLayer(2, SOFTMAX)

    model.add_layer(dense_layer1)
    model.add_layer(dense_layer2)

    batch_size = 3
    rate = 0.5
    model.train_model(data, targets, batch_size, rate)

if __name__ == '__main__':
    # main()
    test_backprop_dense()
