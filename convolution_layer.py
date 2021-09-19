from utils import *
from constant import *
import numpy as np


class ConvolutionLayer:

    def __init__(self,
                 convoFilterSize: 'tuple',
                 activation: 'str',
                 poolingMode: 'str',
                 poolingFilterSize: 'int',
                 convoStride: 'int' = 1,
                 convoPadding: 'int' = 0,
                 poolingStride: 'int' = 1,
                 poolingPadding: 'int' = 0):
        # Early initiate detector stage and pooling stage.
        # NOTE that detector stage and pooling stage won't do anything until we execute their calculate function.
        # Both stages are assigned to None first, add them by using add_detector_stage and add_pooling_stage
        filterSize = convoFilterSize[0]
        nFilter = convoFilterSize[1]

        self.convolution_stage = ConvolutionalStage(
            filterSize, nFilter, convoPadding, convoStride)
        self.detector_stage = DetectorStage(activation)
        self.pooling_stage = PoolingStage(
            poolingFilterSize, poolingPadding, poolingStride, poolingMode)

    def calculate(self, inputs: 'np.ndarray'):
        convoOutput = self.convolution_stage.calculate(inputs)
        detectorOutput = self.detector_stage.calculate(convoOutput)
        poolingOutput = self.pooling_stage.calculate(detectorOutput)

        return poolingOutput


class DetectorStage:

    def __init__(self, det_type: 'str'):
        self.det_type = det_type
        self.channel_output = []

    def get_channel_output(self):
        return self.channel_output

    def detector(self, input: 'np.ndarray'):
        # Type 1 : ReLU
        # Type 2 : Sigmoid

        act_func = self.det_type
        mat_size = len(input)

        new_mat = np.zeros((mat_size, mat_size), dtype=float)

        for i in range(mat_size):
            for j in range(mat_size):
                result = 0
                x = input[i][j]

                if (act_func == RELU):  # Act function is ReLU
                    result = relu(x)
                elif (act_func == SIGMOID):  # Act function is Sigmoid
                    result = sigmoid(x)

                new_mat[i][j] = result

        return new_mat

    def calculate(self, inputs: 'np.ndarray'):  # Looping detector based on how many channels
        featureMaps = []

        for channel_idx in range(len(inputs)):
            featureMaps.append(self.detector(inputs[channel_idx]))

        return np.array(featureMaps)


class PoolingStage:

    def __init__(self, filter_size, mode, padding: 'int' = 0, stride: 'int' = 1):
        self.filter_size = filter_size
        self.padding = padding
        self.stride = stride
        self.mode = mode

    def pooling(self, input):

        # Initiate all vars
        original_mat_size = len(input)
        filter_size = self.filter_size
        padding = self.padding
        stride = self.stride
        mode = self.mode

        # TODO: Prepare the Padding hereW
        matrix = pad2D(input, padding)

        # Update matrix size due to padding
        mat_size = len(matrix)
        result_mat_size = featured_maps_size(
            original_mat_size, filter_size, padding, stride)
        new_mat = np.zeros((result_mat_size, result_mat_size), dtype=float)

        print(result_mat_size)

        # Calculate Pooling
        if (mode == MAX):  # Max Pooling

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

                    # new idx would be integer
                    new_mat_row_idx = int(row_idx / stride)
                    new_mat_col_idx = int(col_idx / stride)

                    new_mat[new_mat_row_idx][new_mat_col_idx] = max_element

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

                    current_avg = current_avg / (filter_size * filter_size)

                    # new idx would be integer
                    new_mat_row_idx = int(row_idx / stride)
                    new_mat_col_idx = int(col_idx / stride)

                    new_mat[new_mat_row_idx][new_mat_col_idx] = current_avg

        return new_mat

    def calculate(self, inputs: 'np.ndarray'):  # Looping pooling based on how many channels
        featureMaps = []
        for channel_idx in range(len(inputs)):
            featureMaps.append(self.pooling(inputs[channel_idx]))

        return np.array(featureMaps)


class ConvolutionalStage:

    def __init__(
        self,
        filterSize: 'int',
        nFilter: 'int',
        paddingSize: 'int',
        strideSize: 'int'
    ) -> None:
        self.inputSize = None
        self.nInput = None
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

    def setInputAttribute(self, inputs: 'np.ndarray'):
        self.nInput = len(inputs)
        self.inputSize = len(inputs[0])

    def calculateFeatureMap(self, inputs: 'np.ndarray', filter: 'np.ndarray', bias: 'np.ndarray'):
        featureMapSize = featured_maps_size(
            self.inputSize, self.filterSize, self.paddingSize, self.strideSize)
        featureMap = np.zeros((featureMapSize, featureMapSize), dtype=float)

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
        self.setInputAttribute(inputs)
        paddedInputs = pad3D(inputs, self.paddingSize)
        featureMaps = []

        for iFilter in range(self.nFilter):
            filter = self.filters[iFilter]
            bias = self.bias[iFilter]
            featureMap = self.calculateFeatureMap(paddedInputs, filter, bias)
            featureMaps.append(featureMap)

        return np.array(featureMaps)
