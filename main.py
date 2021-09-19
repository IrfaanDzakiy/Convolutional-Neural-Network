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
    10, size=(2, 5, 5))


def testConvo():
    tes = ConvolutionalStage(3, 1, paddingSize=1)

    print("INPUTS")
    print(inputs)
    outputs = tes.calculate(inputs)
    print("OUTPUTS")
    print(outputs)

    print()
    print("============== Dense Test =============")
    act_functions = ["relu", "sigmoid", "softmax"]
    dense = Dense_Layer(inputs, 3)
    for i in range(len(act_functions)):
        dense.set_activation_function(act_functions[i])
        output = dense.calculate()
        print(act_functions[i])
        print(output, " Param: ", dense.get_params())


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


def main():
    testConvoLayer()


if __name__ == '__main__':
    main()
