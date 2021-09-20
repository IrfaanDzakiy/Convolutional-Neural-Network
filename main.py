from utils import *
from convolution_layer import *
from dense_layer import *

mat_example = [
    [4, 3, 8, 5],
    [9, 1, 3, 6],
    [6, 3, 5, 3],
    [2, 5, 2, 5]
]

inputs = np.random.randint(
    10, size=(2, 4, 4))


def testConvo():
    tes = ConvolutionalStage(3, 2, paddingSize=1)

    print("INPUTS")
    print(inputs)

    print("FILTER")
    print(tes.filters)

    outputs = tes.calculate(inputs)
    print("OUTPUTS")
    print(outputs)

    newInputs = np.random.randint(
        10, size=(3, 4, 4))

    print("INPUTS")
    print(newInputs)

    print("FILTER")
    print(tes.filters)

    outputs = tes.calculate(newInputs)
    print("OUTPUTS")
    print(outputs)


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
    denseLayer = Dense_Layer(120)
    print("============== Dense Test ===============")
    act_functions = ["relu", "sigmoid", "softmax"]
    for i in range(len(act_functions)):
        denseLayer.set_activation_function(act_functions[i])
        output = denseLayer.calculate(inputs)
        print(act_functions[i])
        print(output, " Param: ", denseLayer.get_params())


def main():
    testConvo()


if __name__ == '__main__':
    main()
