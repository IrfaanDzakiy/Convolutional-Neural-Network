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

        # Belows are the attribute for backprop
        self.learning_rate = None
        self.net_output = None
        self.output = None
        self.delta_weight = None
        self.error_unit = []

    def getName(self):
        return "dense"

    def generate_weight(self):
        return [np.random.random(size=len(self.flattened_input)) for i in range(self.unit)]

    def set_activation_function(self, activation_function):
        self.activation_function = activation_function

    def getParamCount(self):
        return self.params

    def set_input(self, input):
        self.input = np.array(input)
        self.flattened_input: np.array = self.flattened()
        
        if (self.weight == None):
            self.weight: np.array = self.generate_weight()
        
        self.params = len(self.flattened_input) * self.unit

    def getOutputShape(self):
        return (1, self.unit)

    def flattened(self):

        return np.concatenate(([1], self.input.flatten()))

    def normalize(self, arr, t_min=0, t_max=1):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)    
        for i in arr:
            temp = (((i - min(arr))*diff)/diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr

    def calculate(self, inputs):

        output = []
        activated_output = []
        
        # Set input here, including randomize weight
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
            output = self.normalize(output)
            for i in range(len(output)):
                activated_output = softmax(output)
        
        # Insert activated output to class
        self.net_output = output
        self.output = activated_output

        return activated_output

    # Belows are the elements for backward propagation
    
    # Output Backprop
    def D_Ed_Wji_output(self, i, j, target):

        if (self.activation_function == SOFTMAX):
            if (self.output == target):
                return self.output[j] * self.input[i]

            else :
                return (-1) * (1 - self.output[j]) * self.input[i]

        else :
            # Update the error per neuron here
            self.error_unit[j] = self.D_Ed_oj(j, target) * self.D_oj_netj(i,j)

            return self.D_Ed_oj(j, target) * self.D_oj_netj(i,j) * self.D_netj_Wji(i)


    def D_Ed_oj(self, j, target):
        return (-(target - self.output[j]))

    def D_outO_netO(self, j):
        if (self.activation_function == SIGMOID):

            return (self.output[j]) * (1-self.output[j])

        
        elif (self.activation_function == RELU):
            
            return (1 if self.output[j] >= 0 else 0)

        else : # Softmax

            return 

    def D_netO_Wji(self, i):

        return self.input[i]



    # Hidden Backprop
    def D_Ed_Wji(self, prev_layer, next_layer, i, j):
        
        # Also update error of neuron here
        self.error_unit[j] = self.D_Ed_netj(i, j)

        return (self.D_Ed_netj(prev_layer, next_layer, i, j) * self.D_netj_Wji(i, j))

    def D_Ed_netj(self, prev_layer, next_layer, i, j):

        sigma = 0
        # We create the D_Ed_netk_frag and D_netk_oj_frag here
        for next_layer_idx in range(len(next_layer.unit)):
            sigma += next_layer.error_unit[next_layer_idx] * next_layer.weight[next_layer_idx][j]

        error_unit = sigma * self.D_oj_netj(i, j)
        
        self.error_unit[j] = error_unit

        return error_unit * self.D_netj_Wji(i, j)

    def D_Ed_netk_frag(self, next_layer, j): # MAYBE UNUSED
        # If error is 1/2(t-o)^2
        return (-(self.targets - self.output[j]))

    def D_netk_oj_frag(self, next_layer, i, j): # MAYBE UNUSED
        # return sum of weight wkj
        next_layer_weight = next_layer.weight

        curr_sum = 0

        for k in range(len(next_layer_weight)):
            cur_sum = curr_sum + next_layer_weight[k][j]

        return curr_sum

    def D_oj_netj(self, i, j):
        if (self.activation_function == SIGMOID):

            return (self.output[j]) * (1-self.output[j])

        
        elif (self.activation_function == RELU):
            
            return (1 if self.output[j] >= 0 else 0)

        else : # Softmax

            return 


    def D_netj_Wji(self, i, j):
        # return xji
        return self.input[i]


    def train_neurons(self, rate, prev_layer, next_layer, is_output, is_end_of_batch):
        new_delta_weight = self.delta_weight

        # Delta weight is None
        if (new_delta_weight == None):
            new_delta_weight = np.zeros((len(self.unit), len(self.flattened_input)))

        
        if (is_output):
            for neuron_idx in range(len(self.unit)):
                for input_idx in range(len(prev_layer.output)):
                    new_delta_weight[neuron_idx][input_idx] +=  rate * self.D_Ed_Wji_output(input_idx, neuron_idx)

        else :
            for neuron_idx in range(len(self.unit)):
                for input_idx in range(len(prev_layer.output)):
                    new_delta_weight[neuron_idx][input_idx] += rate * self.D_Ed_Wji(prev_layer, next_layer, input_idx, neuron_idx)

        # If at the end of batch, reset delta weight to None and update weight
        if (is_end_of_batch):
            for neuron_idx in range(len(self.unit)):
                for input_idx in range(len(prev_layer.output)):
                    self.weight[neuron_idx][input_idx] += new_delta_weight[neuron_idx][input_idx]
            
            self.delta_weight = None
        
        else :
            self.delta_weight = new_delta_weight
        