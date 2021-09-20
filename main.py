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
    
# def _extract_mnist_images(image_filepath, num_images):
#     _MNIST_IMAGE_SIZE = 28
#     with open(image_filepath, "rb") as f:
#         f.read(16)  # header
#         buf = f.read(_MNIST_IMAGE_SIZE * _MNIST_IMAGE_SIZE * num_images)
#         data = np.frombuffer(
#             buf,
#             dtype=np.uint8,
#         ).reshape(num_images, 1, _MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE)
#         # img = Image.frombuffer('L', (28,28), data)
#         # img.show()
#         return data


def testConvo():
    tes = ConvolutionalStage(5, 6, paddingSize=1)

    print("INPUTS")
    print(inputs)

    print("FILTER")
    print(tes.filters)
    print(tes.getParamCount())

    outputs = tes.calculate(inputs)
    print("OUTPUTS")
    print(outputs)

    print(tes.getParamCount())


def testDetector():
    tes = DetectorStage(SIGMOID)

    print("INPUTS")
    print(inputs)
    outputs = tes.calculate(inputs)
    print("OUTPUTS")
    print(outputs)


def testPooling():
    tes = PoolingStage(2, AVERAGE, stride=2)

    print("INPUTS")
    print(inputs)
    outputs = tes.calculate(inputs)
    print("OUTPUTS")
    print(outputs)


def testConvoLayer():
    convoLayer = ConvolutionLayer((3, 2), RELU, MAX, 2)

    print("INPUTS")
    print(inputs)
    outputs = convoLayer.calculate(inputs)
    print("OUTPUTS")
    print(outputs)


def testDenseLayer():
    denseLayer = DenseLayer(120)
    print("============== Dense Test ===============")
    act_functions = ["relu", "sigmoid", "softmax"]
    for i in range(len(act_functions)):
        denseLayer.set_activation_function(act_functions[i])
        output = denseLayer.calculate(inputs)
        print(act_functions[i])
        print(output, " Param: ", denseLayer.get_params())


def main():
    # print("INPUT")
    # print(inputs)
    # convoLayer = ConvolutionLayer((5, 6), RELU, AVERAGE, 2, poolingStride=2)
    # convoLayer2 = ConvolutionLayer((5, 16), RELU, AVERAGE, 2, poolingStride=2)
    # denseLayer = DenseLayer(120, RELU)
    # denseLayer2 = DenseLayer(84, SIGMOID)
    # denseLayer3 = DenseLayer(100, SOFTMAX)

    inputs = extract_mnist_images("train-images-idx3-ubyte", 1)
    for i in inputs[0]:
        print(i)
    print(np.shape(inputs[0]))
    # convoLayer = ConvolutionLayer((4, 2), RELU, AVERAGE, 1)
    convoLayer = ConvolutionLayer((3, 2), RELU, MAX, 2)
    denseLayer = DenseLayer(120, "sigmoid")
    
    model = Sequential()
    model.add_inputs(inputs[0])
    model.add_layer(convoLayer)
    # model.add_layer(convoLayer2)
    model.add_layer(denseLayer)
    # model.add_layer(denseLayer2)
    # model.add_layer(denseLayer3)
    model.calculate()
    model.print_summary()


if __name__ == '__main__':
    main()
