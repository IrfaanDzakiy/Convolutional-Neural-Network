from utils import *
from convolution_layer import *
from dense_layer import *
from sequential import *
from PIL import Image

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
    
    for image in inputs:
        convoLayer = ConvolutionLayer((3, 2), RELU, MAX, 2)
        denseLayer = DenseLayer(120, "sigmoid")

        model = Sequential()
        model.add_inputs(image)
        model.add_layer(convoLayer)
        model.add_layer(denseLayer)
        model.calculate()
        model.print_summary()


if __name__ == '__main__':
    main()
