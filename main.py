from utils import *
from convolution_layer import *
from dense_layer import *
from sequential import *
from lstm import *
from PIL import Image
import numpy as np
import pandas as pd

mat_example = [
    [4, 3, 8, 5],
    [9, 1, 3, 6],
    [6, 3, 5, 3],
    [2, 5, 2, 5]
]

inputs = np.random.randint(
    256, size=(1, 32, 32))


def main():
    inputs = extract_mnist_images("train-images.idx3-ubyte", 2)
    inputs = convert_grayscale_to_rgb(inputs) / 255

    convoLayer = ConvolutionLayer((3, 2), RELU, MAX, 2, 0.01)
    denseLayer = DenseLayer(10, SOFTMAX)

    model = Sequential()
    model.add_layer(convoLayer)
    model.add_layer(denseLayer)

    image = inputs[0]
    model.add_inputs(image)
    output = model.calculate()

    for o in output:
        print(o.shape)
        print(o)

    model.print_summary()

def main_lstm():
    df = pd.read_csv('bitcoin_price_Training - Training.csv')
    data = df.to_numpy()
    
    #normalize data


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
    epoch = 5
    model.train_model(data, targets, epoch, batch_size, rate)


def csv_convert():
    predictions = [1, 0, 1, 0, 1, 0]
    create_csv(predictions)


def lstm_test():
    inputs = np.array([[1, 2], [0.5, 3]])

    lstm = LSTMLayer(1)
    lstm.calculate(inputs)


def lstm_seq_test():
    inputs = np.array([[1, 2], [0.5, 3]])

    model = Sequential()

    dense_output = DenseLayer(5, SOFTMAX)
    lstm = LSTMLayer(3)

    model.add_layer(lstm)
    model.add_layer(dense_output)

    output = model.calculate()

    print(f"Model Output : \n {output}")


if __name__ == '__main__':
    # main()
    # test_backprop_dense()
    # csv_convert()

    # lstm_test()
    main_lstm()
    # lstm_seq_test()
