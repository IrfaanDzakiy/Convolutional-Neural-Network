from convolution_layer import *
from dense_layer import *


class Sequential:
    def __init__(self):
        self.layers = []
        self.summary = []
        self.inputs = []
        self.outputs = []
        self.targets = None

    def add_inputs(self, inputs):
        self.inputs = inputs

    def add_layer(self, layer):
        self.layers.append(layer)

        curr_total_layers = len(self.layers)
        curr_layer = self.layers[curr_total_layers-1]
        curr_layer.targets = self.targets


    def calculate(self):
        inputs = self.inputs
        i = 1
        for layer in self.layers:
            output_layer = layer.calculate(inputs)
            self.outputs.append(output_layer)

            layer.output.append(output_layer)

            inputs = output_layer
            i += 1

    def print_summary(self):
        convo_count = 0
        dense_count = 0
        print("———————————————————————————————————————————————————————————————————————")
        print("{:<30} {:<30} {:<10}".format(
            'Layer (type) ', 'Output Shape', 'Param #'))
        print("=======================================================================")

        sum_parameter = 0

        for layer in self.layers:
            layerType = layer.getName()
            if(layerType == "convo2D"):
                if(convo_count == 0):
                    postfix = " (Convo2D)"
                else:
                    postfix = "_" + str(convo_count) + " (Convo2D)"
                convo_count += 1
            else:
                if(dense_count == 0):
                    postfix = " (Convo2D)"
                else:
                    postfix = "_" + str(dense_count) + " (Dense)"
                dense_count += 1

            layerTypes = layerType + postfix
            shape = layer.getOutputShape()
            weight = layer.getParamCount()
            sum_parameter += weight
            print("{:<30} {:<30} {:<10}".format(
                layerTypes, str(shape), weight))
            if (layer != self.layers[len(self.layers)-1]):
                print(
                    "———————————————————————————————————————————————————————————————————————")
            else:
                print(
                    "=======================================================================")

        trainable_parameter = sum_parameter
        non_trainable_parameter = sum_parameter - trainable_parameter

        print("Total Params: {}".format(sum_parameter))
        print("Trainable Params: {}".format(trainable_parameter))
        print("Non-trainable Params: {}".format(non_trainable_parameter))
        print()


    def forward_prop(self, data):
        inputs = data
        for layer in self.layers:
            output_layer = layer.calculate(inputs)
            self.outputs.append(output_layer)

            layer.output.append(output_layer)
            inputs = output_layer

        return
    
    # Backprop Dense Elements
    def refresh_error_unit(self):
        for layer_idx in range(len(self.layers)):
            layer.error_unit = []

        return

    def train_model(self, data, targets, batch_size):
        # We always assume data and targets have same length

        data_batch_idx = 0
        for data_idx in range(len(data)):

            curr_data_input = data[data_idx]
            curr_target = targets[data_idx]
            
            # First, we do the forward propagation
            self.forward_prop(data)

            # Backprop steps

            total_layers = len(self.layers)

            is_end_of_batch = False
            if (data_batch_idx == batch_size - 1):
                is_end_of_batch = True

            for i in range(total_layers-1, -1, -1):
                layer = self.layers[i]
                prev_layer = None
                new_prev_weight = None

                # Get prev layer
                if (i == 0):
                    # Create dummy layer
                    dummy_layer = DenseLayer(len(self.inputs),"RELU")
                    dummy_layer.output = self.inputs
                    prev_layer = dummy_layer
                else :
                    prev_layer = self.layers[i-1]

                # Check is output to determine next layer
                is_output = False
                if (i == total_layers - 1):
                    is_output = True
                
                # Update Weight
                if (is_output):
                    layer.train_neurons(prev_layer, None, is_output, is_end_of_batch)
                else :
                    next_layer = self.layers[i+1]
                    layer.train_neurons(prev_layer, next_layer, is_output, is_end_of_batch)    

            # We reset error units in every layers each tuple
            self.refresh_error_unit()

            # Increment data_batch_idx
            if (is_end_of_batch):
                data_batch_idx = 0
            else :
                data_batch_idx += 1

        return 