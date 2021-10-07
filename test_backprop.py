import numpy as np
from convolution_layer import PoolingStage
from constant import *

input: 'np.ndarray' = np.arange(32)
input = input.reshape((2, 4, 4))

print(input)

pooling = PoolingStage(3, AVERAGE, stride=1)
output = pooling.calculate(input)

print(output)

backprop = pooling.backprop(output)

print(backprop)
