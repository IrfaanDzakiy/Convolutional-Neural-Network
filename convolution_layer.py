from utils import *
import numpy as np


class ConvolutionLayer:

    def __init__(self, lay_num):
        # Early initiate detector stage and pooling stage.
        # NOTE that detector stage and pooling stage won't do anything until we execute their calculate function.
        # Both stages are assigned to None first, add them by using add_detector_stage and add_pooling_stage

        self.layer_number = lay_num
        self.detector_stage = None
        self.pooling_stage = None

    def add_detector_stage(self, channel_input, def_type):
        self.detector_stage = DetectorStage(channel_input, def_type)

    def add_pooling_stage(self, channel_input, filter_size, padding, stride, mode):
        self.pooling_stage = PoolingStage(
            channel_input, filter_size, padding, stride, mode)


class DetectorStage:

    def __init__(self, channel_input, det_type):
        self.channel_input = channel_input
        self.det_type = det_type
        self.channel_output = []

    def get_channel_output(self):
        return self.channel_output

    def detector(self, channel_idx):
        # Type 1 : ReLU
        # Type 2 : Sigmoid

        matrix = self.channel_input[channel_idx]

        act_func = self.det_type
        mat_size = len(matrix)

        new_mat = [[] for i in range(mat_size)]

        for i in range(mat_size):
            for j in range(mat_size):

                result = 0
                x = matrix[i][j]

                if (act_func == 1):  # Act function is ReLU
                    result = relu(x)
                else:  # Act function is Sigmoid
                    result = sigmoid(x)

                new_mat[i].append(result)

        return new_mat

    def calculate(self):  # Looping detector based on how many channels

        for channel_idx in range(len(self.channel_input)):
            self.channel_output.append(self.detector(channel_idx))

        return


class PoolingStage:

    def __init__(self, channel_input, filter_size, padding, stride, mode):
        self.channel_input = channel_input
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
        self.mode = mode
        self.channel_output = []

    def get_channel_output(self):
        return self.channel_output

    def pooling(self, channel_idx):

        # Initiate all vars
        matrix = self.channel_input[channel_idx]

        original_mat_size = len(matrix)
        filter_size = self.filter_size
        padding = self.padding
        stride = self.stride
        mode = self.mode

        # TODO: Prepare the Padding here
        padded_mat = [[]]
        if (padding != 0):

            # Add Top and Bottom padding
            for i in range(padding):
                pad_row = [0 for j in range(original_mat_size)]
                matrix.insert(0, pad_row)
                matrix.insert(len(matrix), pad_row)

            # Add Side Padding
            for i in range(len(matrix)):
                curr_row = matrix[i]
                for j in range(padding):
                    curr_row.insert(0, 0)
                    curr_row.insert(len(curr_row), 0)

                matrix[i] = curr_row

            print("Done doing Padding")

        # Update matrix size due to padding
        mat_size = len(matrix)
        result_mat_size = featured_maps_size(
            original_mat_size, filter_size, padding, stride)
        new_mat = [[] for i in range(result_mat_size)]

        print(result_mat_size)

        # Calculate Pooling
        if (mode == 1):  # Max Pooling

            for row_idx in range(0, mat_size, stride):
                for col_idx in range(0, mat_size, stride):

                    if (row_idx + filter_size > mat_size):  # Horizontal border
                        continue

                    if (col_idx + filter_size > mat_size):  # Vertical border
                        continue

                    max_element = matrix[row_idx][col_idx]
                    for i in range(row_idx, row_idx + filter_size):
                        for j in range(col_idx, col_idx + filter_size):
                            if (max_element < matrix[i][j]):
                                max_element = matrix[i][j]

                            print(matrix[i][j])

                    print()

                    # new idx would be integer
                    new_mat_row_idx = int(row_idx / stride)
                    new_mat_col_idx = int(col_idx / stride)

                    new_mat[new_mat_row_idx].append(max_element)

            return new_mat

        else:  # Average Pooling

            for row_idx in range(0, mat_size, stride):
                for col_idx in range(0, mat_size, stride):

                    if (row_idx + filter_size > mat_size):  # Horizontal border
                        continue

                    if (col_idx + filter_size > mat_size):  # Vertical border
                        continue

                    current_avg = 0
                    for i in range(row_idx, row_idx + filter_size):
                        for j in range(col_idx, col_idx + filter_size):
                            current_avg = current_avg + matrix[i][j]

                            print(matrix[i][j])

                    print()

                    current_avg = current_avg / (filter_size * filter_size)

                    # new idx would be integer
                    new_mat_row_idx = int(row_idx / stride)
                    new_mat_col_idx = int(col_idx / stride)

                    new_mat[new_mat_row_idx].append(current_avg)

            return new_mat

    def calculate(self):  # Looping pooling based on how many channels

        for channel_idx in range(len(self.channel_input)):
            self.channel_output.append(self.pooling(channel_idx))

        return


class ConvolutionalStage:

    def __init__(
        self,
        inputSize: 'int',
        nInput: 'int',
        filterSize: 'int',
        nFilter: 'int',
        paddingSize: 'int' = 0,
        strideSize: 'int' = 1,
    ) -> None:
        self.inputSize = inputSize
        self.nInput = nInput
        self.paddingSize = paddingSize
        self.strideSize = strideSize
        self.filterSize = filterSize
        self.nFilter = nFilter
        self.filters = self.generateParams()
        self.bias = self.generateBias()

    def generateParams(self):
        return np.random.randint(
            10, size=(self.nFilter, self.filterSize, self.filterSize))

    def generateBias(self):
        return np.full((self.nFilter, 1), 1)

    def pad(self, inputs: 'np.ndarray'):
        paddedInputs = []
        padding_dim = [(self.paddingSize, self.paddingSize),
                       (self.paddingSize, self.paddingSize)]
        for i in range(len(inputs)):
            paddedInputs.append(
                np.pad(inputs[i], padding_dim, mode='constant'))
        return np.array(paddedInputs)

    def calculateFeatureMap(self, inputs: 'np.ndarray', filter: 'np.ndarray', bias: 'np.ndarray'):
        featureMapSize = featured_maps_size(
            self.inputSize, self.filterSize, self.paddingSize, self.strideSize)
        featureMap = np.zeros((featureMapSize, featureMapSize), dtype=int)

        for iInput in range(self.nInput):
            input = inputs[iInput]
            for i in range(0, featureMapSize, self.strideSize):
                for j in range(0, featureMapSize, self.strideSize):
                    inputSubset = input[i:i +
                                        self.filterSize, j:j + self.filterSize]
                    featureMap[i][j] += np.sum(
                        np.multiply(inputSubset, filter))

        featureMap = featureMap + bias
        return featureMap

    def calculate(self, inputs: 'np.ndarray'):
        paddedInputs = self.pad(inputs)
        featureMaps = []

        for iFilter in range(self.nFilter):
            filter = self.filters[iFilter]
            bias = self.bias[iFilter]
            featureMap = self.calculateFeatureMap(paddedInputs, filter, bias)
            featureMaps.append(featureMap)

        return np.array(featureMaps)
