from utils import *
from constant import *
import numpy as np


class ConvolutionLayer:
    def __init__(self,
                 convoFilterSize: 'tuple',
                 activation: 'str',
                 poolingMode: 'str',
                 poolingFilterSize: 'int',
                 learn_rate: 'float',
                 inputShape: 'tuple' = None,
                 convoStride: 'int' = None,
                 convoPadding: 'int' = None,
                 poolingStride: 'int' = None,
                 poolingPadding: 'int' = None):
        # Early initiate detector stage and pooling stage.
        # NOTE that detector stage and pooling stage won't do anything until we execute their calculate function.
        # Both stages are assigned to None first, add them by using add_detector_stage and add_pooling_stage
        filterSize = convoFilterSize[0]
        nFilter = convoFilterSize[1]

        self.convolution_stage = ConvolutionalStage(
            filterSize, nFilter, convoPadding, convoStride)
        self.detector_stage = DetectorStage(activation)
        self.pooling_stage = PoolingStage(
            poolingFilterSize, poolingMode, poolingPadding, poolingStride)
        self.inputShape = inputShape
        self.learn_rate = learn_rate

        if (inputShape is not None):
            self.setInputShape(inputShape)

        self.targets = None
        self.output = []
        self.error_unit = []

    def getName(self):
        return "convo2D"

    def getParamCount(self):
        return self.convolution_stage.getParamCount()

    def setInputShape(self, shape: 'tuple'):
        # Here we are propagating the input shape of the layer to each stage
        self.convolution_stage.setInputShape(shape)
        convoOutput = self.convolution_stage.getOutputShape()

        # Because Detector Stage is input/output dependant, we dont need to set the input shape
        self.pooling_stage.setInputShape(convoOutput)

    def getOutputShape(self):
        return self.pooling_stage.getOutputShape()

    def calculate(self, inputs: 'np.ndarray'):
        convoOutput = self.convolution_stage.calculate(inputs)
        detectorOutput = self.detector_stage.calculate(convoOutput)
        poolingOutput = self.pooling_stage.calculate(detectorOutput)

        return poolingOutput

    def backprop(self, dL_dOut: 'np.ndarray'):
        dL_dDetector = self.pooling_stage.backprop(dL_dOut)
        dL_dConvo = self.detector_stage.backprop(dL_dDetector)
        dL_dIn = self.convolution_stage.backprop(dL_dConvo, self.learn_rate)

        return dL_dIn


class DetectorStage:

    def __init__(self, det_type: 'str'):
        self.det_type = det_type
        self.channel_output = []
        self.input: 'np.ndarray' = None
        self.nInput: 'int' = None
        self.inputSize: 'int' = None

    def get_channel_output(self):
        return self.channel_output

    def setInput(self, inputs: 'np.ndarray'):
        self.input = inputs
        self.nInput = inputs.shape[0]
        self.inputSize = inputs.shape[1]

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
        self.setInput(inputs)
        featureMaps = []

        for channel_idx in range(len(inputs)):
            featureMaps.append(self.detector(inputs[channel_idx]))

        return np.array(featureMaps)

    def backprop(self, dL_dOut: 'np.ndarray'):
        inputs = self.input.copy()
        if (self.det_type == RELU):
            dOut_dIn = dRelu(inputs)
        elif (self.det_type == SIGMOID):
            dOut_dIn = dSigmoid(inputs)

        return dL_dOut * dOut_dIn


class PoolingStage:

    def __init__(self, filter_size: 'int', mode: 'str', padding: 'int' = None, stride: 'int' = None):
        self.filter_size = filter_size
        self.padding = padding if padding != None else 1
        self.stride = stride if stride != None else filter_size
        self.mode = mode
        self.nInput = None
        self.inputSize = None
        self.input: 'np.ndarray' = None

    def setInput(self, inputs: 'np.ndarray'):
        self.input = inputs
        self.nInput = inputs.shape[0]
        self.inputSize = inputs.shape[1]

    def getOutputShape(self):
        if (self.nInput is None or self.inputSize is None):
            return (0, 0, 0)

        featureMapSize = featured_maps_size(
            self.inputSize, self.filter_size, self.padding, self.stride)
        print()

        return (self.nInput, featureMapSize, featureMapSize)

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
        self.setInput(inputs)
        featureMaps = []
        for channel_idx in range(len(inputs)):
            featureMaps.append(self.pooling(inputs[channel_idx]))

        return np.array(featureMaps)

    def backprop(self, dL_dOut: 'np.ndarray'):
        (n_output, size_output, _) = self.getOutputShape()
        paddedInput = pad3D(self.input, self.padding)
        dL_dInput = np.zeros(paddedInput.shape)

        for i_dOut in range(n_output):
            for i in range(0, size_output):
                for j in range(0, size_output):
                    rec_i_start = i * self.stride
                    rec_i_end = rec_i_start + self.filter_size
                    rec_j_start = j * self.stride
                    rec_j_end = rec_j_start + self.filter_size

                    if (self.mode == MAX):
                        rec_field = paddedInput[i_dOut,
                                                rec_i_start:rec_i_end, rec_j_start:rec_j_end]
                        (x_max, y_max) = np.argwhere(
                            rec_field == dL_dOut[i_dOut, i, j])[0]
                        dL_dInput[i_dOut, rec_i_start + x_max, rec_j_start +
                                  y_max] = dL_dOut[i_dOut, i, j]
                    elif (self.mode == AVERAGE):
                        dL_dInput[i_dOut, rec_i_start:rec_i_end,
                                  rec_j_start:rec_j_end] += dL_dOut[i_dOut, i, j]/(self.filter_size**2)

        if (self.padding > 0):
            dL_dInput = dL_dInput[:, self.padding:-
                                  self.padding, self.padding:-self.padding]
        return dL_dInput


class ConvolutionalStage:

    def __init__(
        self,
        filterSize: 'int',
        nFilter: 'int',
        paddingSize: 'int' = None,
        strideSize: 'int' = None
    ) -> None:
        self.input: 'np.ndarray' = None
        self.inputSize: 'int' = None
        self.nInput: 'int' = None
        self.paddingSize = paddingSize if paddingSize != None else 0
        self.strideSize = strideSize if strideSize != None else 1
        self.filterSize = filterSize
        self.nFilter = nFilter
        self.filters: 'np.ndarray' = None
        self.bias: 'np.ndarray' = self.generateBias()

    def getParamCount(self):
        if (self.nInput is None):
            return 0
        return self.nFilter * ((self.filterSize * self.filterSize * self.nInput) + 1)

    def generateParams(self):
        return np.random.randn(self.nFilter, self.nInput, self.filterSize, self.filterSize)

    def generateBias(self):
        return np.full((self.nFilter, 1), 1)

    def setInput(self, inputs: 'np.ndarray'):
        self.input = inputs
        self.nInput = inputs.shape[0]
        self.inputSize = inputs.shape[1]
        self.filters = self.generateParams()

    def getOutputShape(self):
        featureMapSize = 0
        if (self.inputSize is not None):
            featureMapSize = featured_maps_size(
                self.inputSize, self.filterSize, self.paddingSize, self.strideSize)

        return (self.nFilter, featureMapSize, featureMapSize)

    def convolve(self, inputs: 'np.ndarray', filter: 'np.ndarray', bias: 'np.ndarray'):
        featureMapSize = featured_maps_size(
            self.inputSize, self.filterSize, self.paddingSize, self.strideSize)
        featureMap = np.zeros((featureMapSize, featureMapSize), dtype=float)

        for iInput in range(self.nInput):
            input = inputs[iInput]
            inputFilter = filter[iInput]
            for i in range(0, featureMapSize, self.strideSize):
                for j in range(0, featureMapSize, self.strideSize):
                    inputSubset = input[i:i +
                                        self.filterSize, j:j + self.filterSize]
                    featureMap[i][j] += np.sum(
                        np.multiply(inputSubset, inputFilter))

        featureMap = featureMap + bias
        return featureMap

    def calculate(self, inputs: 'np.ndarray'):
        oldInput = self.input
        if (self.nInput is None or self.inputSize is None or oldInput.shape != inputs.shape):
            self.setInput(inputs)
        paddedInputs = pad3D(inputs, self.paddingSize)
        featureMaps = []

        for iFilter in range(self.nFilter):
            filter = self.filters[iFilter]
            bias = self.bias[iFilter]
            featureMap = self.convolve(paddedInputs, filter, bias)
            featureMaps.append(featureMap)

        return np.array(featureMaps)

    def backprop(self, dL_dOut: 'np.ndarray', learn_rate: 'float'):
        paddedInput = pad3D(self.input, self.paddingSize)
        dL_dFilters = np.zeros(self.filters.shape)
        dL_dB = np.zeros(self.bias.shape)
        dL_dIn = np.zeros(paddedInput.shape)
        dL_dOut = np.reshape(
            dL_dOut, (dL_dOut.shape[0], 1, dL_dOut[1], dL_dOut[2]))
        dummy_bias = np.zeros(self.bias.shape)

        for iFilter in range(self.nFilter):
            filter = dL_dOut[iFilter]
            bias = dummy_bias[iFilter]
            dL_dFilters[iFilter] = self.convolve(paddedInput, filter, bias)
            dL_dB[iFilter] = np.sum(filter)

        self.filters -= learn_rate * dL_dFilters
        self.bias -= learn_rate * dL_dB

        dL_dIn = dL_dIn if (
            self.paddingSize <= 0) else dL_dIn[:, self.paddingSize:-self.paddingSize, self.paddingSize:-self.paddingSize]
        return dL_dIn
