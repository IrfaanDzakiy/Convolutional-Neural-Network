import numpy as np
from PIL import Image
from convolution_layer import *
from sequential import *
from dense_layer import *
from constant import *
from utils import *

if __name__ == '__main__':
    n_data = 1000

    # x_train = extract_mnist_images("train-images.idx3-ubyte", n_data)
    # x_train = convert_to_grayscale(x_train) / 255
    # y_train = extract_mnist_labels("train-labels.idx1-ubyte", n_data)

    x = extract_mnist_images("train-images.idx3-ubyte", n_data)
    x = convert_to_grayscale(x) / 255
    y = extract_mnist_labels("train-labels.idx1-ubyte", n_data)

    x_validation, x_train = cross_validation(10, x)
    y_validation, y_train = cross_validation(10, y)

    y_train = one_hot_encoder(y_train)

    print("Image")
    print(x_train.shape)

    print("Label")
    print(y_train.shape)

    print("X :", x.shape)
    print("Y :", y.shape)
    print("X train: ", x_train.shape)
    print("Y train: ", y_train.shape)
    print("X validation: ", x_validation.shape)
    print("Y validation: ", y_validation.shape)

    model = Sequential()

    convo_layer = ConvolutionLayer((5, 4), SIGMOID, MAX, 2)
    dense_layer1 = DenseLayer(10, SOFTMAX)

    model.add_layer(convo_layer)
    model.add_layer(dense_layer1)

    batch_size = 5
    epoch = 3
    rate = 0.3
    model.train_model(x_train, y_train, epoch, batch_size, rate)

    y_predictions = []

    for image in x_validation:
        output = np.argmax(model.forward_prop(image))
        y_predictions.append(output)

    y_predictions = np.array(y_predictions)
    print("Y predictions :", y_predictions)
    print("Y validations :", y_validation)
    print("Accuracy :", accuracy(y_predictions,y_validation))
    
    create_csv(y_predictions)
