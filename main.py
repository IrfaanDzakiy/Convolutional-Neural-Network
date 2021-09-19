from utils import *
from convolution_layer import *
from dense_layer import *

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
    
    print()
    print("============== Dense Test =============")
    act_functions = ["relu", "sigmoid", "softmax"]
    dense = Dense_Layer(inputs, 3)
    for i in range(len(act_functions)):
        dense.set_activation_function(act_functions[i])
        output = dense.calculate()
        print(act_functions[i])
        print(output, " Param: ", dense.get_params())
    
    

if __name__ == '__main__':
    main()
