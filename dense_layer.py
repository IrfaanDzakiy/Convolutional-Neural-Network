from constant import RELU, SIGMOID, SOFTMAX
from utils import *
import numpy as np


class DenseLayer:
    def __init__(self, unit, activation_function: 'str'):
        self.unit = unit
        self.activation_function = activation_function
        self.input = None
        self.flattened_input = None
        self.weight = None
        self.params = None

    def getName(self):
        return "dense"

    def generate_weight(self):
        return [np.random.randint(10, size=len(self.flattened_input)) for i in range(self.unit)]

    def set_activation_function(self, activation_function):
        self.activation_function = activation_function

    def getParamCount(self):
        return self.params

    def set_input(self, input):
        self.input = np.array(input)
        self.flattened_input: np.array = self.flattened()
        self.weight: np.array = self.generate_weight()
        self.params = len(self.flattened_input) * self.unit

    def getOutputShape(self):
        return (1, self.unit)

    def flattened(self):

        return np.concatenate(([1], self.input.flatten()))

    def calculate(self, inputs):

        output = []
        activated_output = []
        
        self.set_input(inputs)

        for i in range(self.unit):
            # Sum Product
            output.append(np.dot(self.flattened_input, self.weight[i]))

        if self.activation_function == RELU:
            for i in range(len(output)):
                activated_output.append(relu(output[i]))
        elif self.activation_function == SIGMOID:
            for i in range(len(output)):
                activated_output.append(sigmoid(output[i]))
        elif self.activation_function == SOFTMAX:
            for i in range(len(output)):
                activated_output = softmax(output)

        return activated_output
