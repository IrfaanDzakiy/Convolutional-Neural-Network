import numpy as np
from convolution_layer import PoolingStage, DetectorStage
from constant import *

input: 'np.ndarray' = np.random.randn(2, 4, 4)

print(input)

pooling = PoolingStage(2, MAX)
detector = DetectorStage(SIGMOID)

print("Fowardprop Layer 1")
output1 = detector.calculate(input)
print(output1, end='\n\n')

print("Fowardprop Layer 2")
output2 = pooling.calculate(output1)
print(output2, end='\n\n')

print("Backprop Layer 2")
backprop2 = pooling.backprop(output2)
print(backprop2, end='\n\n')

print("Backprop Layer 1")
backprop1 = detector.backprop(backprop2)
print(backprop1, end='\n\n')
