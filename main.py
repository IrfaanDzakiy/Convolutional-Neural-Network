from utils import *
from convolution_layer import *
from dense_layer import *
from sequential import *

mat_example = [
    [4, 3, 8, 5],
    [9, 1, 3, 6],
    [6, 3, 5, 3],
    [2, 5, 2, 5]
]

inputs = np.random.randint(
    256, size=(1, 32, 32))


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
    print("INPUT")
    print(inputs)
    convoLayer = ConvolutionLayer((5, 6), RELU, AVERAGE, 2, poolingStride=2)
    convoLayer2 = ConvolutionLayer((5, 16), RELU, AVERAGE, 2, poolingStride=2)
    denseLayer = DenseLayer(120, RELU)
    denseLayer2 = DenseLayer(84, RELU)
    denseLayer3 = DenseLayer(10, SOFTMAX)

    model = Sequential()
    model.add_inputs(inputs)
    model.add_layer(convoLayer)
    model.add_layer(convoLayer2)
    model.add_layer(denseLayer)
    model.add_layer(denseLayer2)
    # model.add_layer(denseLayer3)
    model.calculate()
    model.print_summary()


if __name__ == '__main__':
    main()
