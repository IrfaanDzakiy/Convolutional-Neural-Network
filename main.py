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


def test():

    inputs = [i for i in range(20)]
    targets = []

    model = Sequential()
    

    dense_layer = DenseLayer(10, "sigmoid")
    dense_layer.flattened_input = inputs
    x = dense_layer.generate_weight()
    print(len(x))

if __name__ == '__main__':
    # main()
    test()
