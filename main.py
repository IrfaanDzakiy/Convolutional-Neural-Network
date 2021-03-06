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

# store min max val per col
minMax = []
    
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
    # df = pd.read_csv('bitcoin_price_1week_Test - Test.csv')
    df = pd.read_csv('bitcoin_price_Training - Training.csv')

    
    data = df.drop(columns=["Date"])
    data = data.head(32)
    data = data.iloc[::-1]
    # data = data.drop(columns=["Volume", "Market Cap"])
    
    data['Volume'] = data['Volume'].str.replace(',', '')
    data['Market Cap'] = data['Market Cap'].str.replace(',', '')
  
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
    data['Market Cap'] = pd.to_numeric(data['Market Cap'], errors='coerce')
        
    data["Open"] = data["Open"].astype(float)
    data["High"] = data["High"].astype(float)
    data["Low"] = data["Low"].astype(float)
    data["Close"] = data["Close"].astype(float)
    data = data.to_numpy()
    print(data.shape)
    print(data)

    # normalize data
    x, y = data.shape

    # normalized_data = np.zeros((x,y))
    normalized_data = [[None for j in range(y)] for i in range(x)]
    
    for j in range(y):
        xMin = None
        xMax = None
        for i in range(x):
            if xMin == None and xMax == None:
                xMin, xMax = getMinMaxVal(data, x, j)
                minMax.append([xMin, xMax])
                # print('minMax', minMax)
                print('xMin = ', xMin)
                print('xMax = ', xMax)

            if data[i][j] == '-':
                normalized_data[i][j] = minMaxScaler(xMin, xMin, xMax)
            else:
                normalized_data[i][j] = minMaxScaler(data[i][j], xMin, xMax)

    # print(np.array(normalized_data))
    return np.asarray(normalized_data, dtype='float64')



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
    inputs = np.array([[1, 2, 3, 4, 5, 6], [0.5, 3, 3, 4, 5, 6], [0.5, 3, 3, 4, 5, 6]])
    print(inputs.shape)
    lstm = LSTMLayer(1)
    lstm.calculate(inputs)


def lstm_seq_test():
    inputs = np.array([[1, 0.4, 0.3], [0.5, 0.2, 0.1]])
    inputs = main_lstm()
    # print(inputs)
    model = Sequential()

    dense_output = DenseLayer(3, RELU)
    lstm = LSTMLayer(5)

    model.add_layer(lstm)
    model.add_layer(dense_output)

    output = model.forward_prop(inputs)
    model.print_summary()
    
    denorm_output = []
    
    print(minMax)
    for i in range(len(output)):
        denorm_output.append(reverseMinMaxScaler(output[i], minMax[i][0], minMax[i][1]))

    print(f"Model Output : \n {denorm_output}")
    
    
    


if __name__ == '__main__':
    # main()
    # test_backprop_dense()
    # csv_convert()

    # lstm_test()
    # main_lstm()
    lstm_seq_test()
