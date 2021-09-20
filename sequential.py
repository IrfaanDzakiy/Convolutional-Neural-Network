from convolution_layer import *
from dense_layer import *

class Sequential:
    def __init__(self):
        self.layers = []
        self.summary = []
        self.inputs = []
        self.outputs = []
    
    def add_inputs(self, inputs):
      self.inputs = inputs
      
    def add_layer(self, layer):
      self.layers.append(layer)
    
    
    def calculate(self):
      inputs = self.inputs
      
      for layer in self.layers:
        output_layer = layer.calculate(inputs)
        self.outputs.append(output_layer)
        
        inputs = output_layer
    
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
        if(layerType=="convo2D"):
          if(convo_count==0):
            postfix = " (Convo2D)"
          else:
            postfix = "_" + convo_count + " (Convo2D)"
          convo_count+=1
        else:
          if(dense_count==0):
            postfix = " (Convo2D)"
          else: 
            postfix = "_" + dense_count + " (Dense)"
          dense_count+=1
          
        layerTypes = layerType + postfix
        shape = layer.getOutputShape()
        weight = layer.getParamCount()
        sum_parameter += weight
        print("{:<30} {:<30} {:<10}".format(layerTypes, str(shape), weight))
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