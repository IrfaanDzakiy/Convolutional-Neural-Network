from utils import *
from convolution_layer import *

mat_example = [
    [4, 3, 8, 5],
    [9, 1, 3, 6],
    [6, 3, 5, 3],
    [2, 5, 2, 5]
]


def main():
    tes = ConvolutionalStage(3, 1, paddingSize=1)
    inputs = np.random.randint(
        10, size=(2, 5, 5))
    print("INPUTS")
    print(inputs)
    outputs = tes.calculate(inputs)
    print("OUTPUTS")
    print(outputs)


if __name__ == '__main__':
    main()
