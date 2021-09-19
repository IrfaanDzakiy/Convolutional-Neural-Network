from utils import *
import numpy as np

class Dense_Layer:
    def __init__(self, input, unit, activation_function=None):
        self.input = np.array(input)
        self.unit = unit
        self.flattened_input: np.array = np.concatenate(([1], self.input.flatten()))
        self.weight: np.array = self.generate_weight()
        self.activation_function = activation_function
        self.params = len(self.flattened_input) * self.unit
        
    def generate_weight(self):
        
        return [np.random.randint(10, size=len(self.flattened_input)) for i in range(self.unit)]
    
    def set_activation_function(self, activation_function):
        self.activation_function = activation_function
    
    def get_params(self):
        return self.params
    
    def calculate(self):
        
        # print("FLATTENED INPUT")
        # print(self.flattened_input)
        # print("WEIGHT")
        # print(self.weight)
        
        output = []
        activated_output = []
        
        for i in range(self.unit):
            # Sum Product
            output.append(np.dot(self.flattened_input, self.weight[i]))
        
        
        if self.activation_function.lower() == "relu":
            for i in range(len(output)):
                activated_output.append(relu(output[i]))
        elif self.activation_function.lower() == "sigmoid":
            for i in range(len(output)):
                activated_output.append(sigmoid(output[i]))
        elif self.activation_function.lower() == "softmax":
            for i in range(len(output)):
                activated_output = softmax(output)
                
        return activated_output
    
    

