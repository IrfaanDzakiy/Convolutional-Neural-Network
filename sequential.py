from convolution_layer import *
from dense_layer import *
import pickle


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

    # DEPRECATED
    def calculate(self):
        inputs = self.inputs
        i = 1
        for layer in self.layers:
            print("layer ke", i)
            output_layer = layer.calculate(inputs)
            self.outputs.append(output_layer)
            layer.output.append(output_layer)

            inputs = output_layer
            i += 1
        return self.outputs

    def save_model(self, filename):
        i = 1
        file = open(filename, 'wb')

        data_dict = {}
        for layer in self.layers:
            data_dict["layer" + str(i)] = layer.get_data()
            i += 1

        pickle.dump(data_dict, file)
        file.close()
        print("MODEL SAVED")

    def load_model(self, filename):

        file = open(filename, 'rb')

        data = pickle.load(file)

        file.close()
        i = 0
        for key, layer in data.items():
            type = layer['type']

            if type == "DENSE":
                self.add_layer(DenseLayer(
                    layer['unit'], layer['activation_function']))
                self.layers[i].set_weight(layer['weight'])
                self.layers[i].set_params(layer['params'])

            elif type == "CONVOLUTION":
                self.add_layer(
                    ConvolutionLayer(
                        layer['convolution_filter_size'],
                        layer['detector_activation_function'],
                        layer['pooling_mode'],
                        layer['pooling_filter_size'],
                        layer['convolution_stride'],
                        layer['convolution_padding'],
                        layer['pooling_stride'],
                        layer['pooling_padding']
                    ))
                self.layers[i].set_kernel(layer["convolution_kernel"])
                self.layers[i].set_output_shape(layer["output_shape"])
                self.layers[i].set_params(layer["params"])

            i += 1

        print("MODEL LOADED")

    def print_summary(self):
        convo_count = 0
        dense_count = 0
        lstm_count = 0
        print("———————————————————————————————————————————————————————————————————————")
        print("{:<30} {:<30} {:<10}".format(
            'Layer (type) ', 'Output Shape', 'Param #'))
        print("=======================================================================")
        i = 1
        sum_parameter = 0
        for layer in self.layers:
            layerType = layer.getName()
            if(layerType == "convo2D"):
                if(convo_count == 0):
                    postfix = " (Convo2D)"
                else:
                    postfix = "_" + str(convo_count) + " (Convo2D)"
                convo_count += 1
            elif (layerType == "dense"):
                if(dense_count == 0):
                    postfix = " (Dense)"
                else:
                    postfix = "_" + str(dense_count) + " (Dense)"
                dense_count += 1
            elif (layerType == 'lstm'):
                if (lstm_count == 0):
                    postfix = " (LSTM)"
                else:
                    postfix = "_" + str(lstm_count) + " (LSTM)"

            layerTypes = layerType + postfix
            shape = layer.getOutputShape()
            weight = layer.getParamCount()
            sum_parameter += weight
            i += 1
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

            # Masukin hasil aktivasi ke output dari suatu layer
            layer.output = output_layer

            # Output as new input
            inputs = output_layer
        return inputs

    def update_convo_params(self):
        for layer in self.layers:
            if (layer.getName() == 'convo2D'):
                layer.updateParams()

    # Backprop Dense Elements
    def refresh_error_unit(self):
        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            if (layer.getName() == 'dense'):
                layer.error_unit = [0 for i in range(layer.unit)]

    def train_model(self, data, targets, epoch, batch_size, rate):

        # We always assume data and targets have same length

        for e in range(epoch):
            data_batch_idx = 0
            for data_idx in range(len(data)):

                curr_data_input = data[data_idx]
                curr_target = targets[data_idx]

                # First, we do the forward propagation
                self.add_inputs(curr_data_input)
                self.forward_prop(curr_data_input)

                # Backprop steps

                total_layers = len(self.layers)

                is_end_of_batch = False
                if (data_batch_idx == batch_size - 1):
                    is_end_of_batch = True

                elif (data_idx == len(data) - 1):
                    is_end_of_batch = True

                # Train neurons per layer
                learning_result = None
                for i in range(total_layers-1, -1, -1):
                    layer = self.layers[i]

                    # Check is output to determine next layer
                    is_output = False
                    if (i == total_layers - 1):
                        is_output = True

                    if (is_output):
                        learning_result = layer.train_neurons(
                            rate, None, curr_target, is_output, is_end_of_batch)
                    else:
                        if (layer.getName() == 'dense'):
                            next_layer = self.layers[i+1]
                            learning_result = layer.train_neurons(
                                rate, next_layer, curr_target, is_output, is_end_of_batch)
                        elif (layer.getName() == 'convo2D'):
                            learning_result = layer.backprop(
                                learning_result, rate)

                # We reset error units in every layers each tuple
                self.refresh_error_unit()

                # Increment data_batch_idx
                if (is_end_of_batch):
                    self.update_convo_params()
                    data_batch_idx = 0
                    print("BATCH BERAKHIR, WEIGHT ADALAH")
                    # self.print_layers_weight()
                else:
                    data_batch_idx += 1

                print(f"DATA KE-{data_idx+1} TELAH DILEWATI")

            print(f"EPOCH {e} TELAH DILEWATI")

        # Check the weight here
        print("WEIGHT AKHIR")
        self.print_layers_weight()

    def print_layers_weight(self):
        for layer in self.layers:
            if (layer.getName() == 'dense'):
                print("DENSE LAYER WEIGHTS")
                print(layer.weight)
            else:
                print("CONVO LAYER KERNELS")
                print(layer.getParams())
