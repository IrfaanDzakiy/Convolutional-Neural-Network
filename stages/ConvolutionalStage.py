class ConvolutionalStage:

  def __init__(
    self, 
    inputSize: 'int', 
    paddingSize: 'int', 
    strideSize: 'int',
    filterSize: 'int',
    nFilter: 'int') -> None:
      self.inputSize = inputSize
      self.paddingSize = paddingSize
      self.strideSize = strideSize
      self.filterSize = filterSize
      self.nFilter = nFilter

