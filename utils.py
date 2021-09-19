import math

def relu(x):
    return max(0, x)

def sigmoid(x):
    return (1 / (1 + math.exp(-x)))

def softmax(input):
    result = []
    sum_of_exp_z = 0
    for i in range(len(input)):
        sum_of_exp_z += math.exp(input[i])
    
    for i in range(len(input)):
        result.append(math.exp(input[i]) / sum_of_exp_z)
    return result
    
def featured_maps_size(matrix_size, filter_size, padding, stride):

    return int(((matrix_size - filter_size + (2 * padding)) / stride) + 1)


def print_matrix(matrix):
    mat_len = len(matrix)

    for i in range(mat_len):
        for j in range(mat_len):
            print(f"{matrix[i][j]}", end =" ")
        print("")
## Python program to print the data

def sum_parameter(d):
  sum_params = 0
  for k, v in d.items():
    layer, shape, param = v
    sum_params += param
  
  return sum_params

def print_layer(d):
  print()
  print("———————————————————————————————————————————————————————————————————————")
  print("{:<30} {:<30} {:<10}".format('Layer (type) ','Output Shape','Param #'))
  print("=======================================================================")
  for k, v in d.items():
    lang, perc, change = v
    print("{:<30} {:<30} {:<10}".format(lang, perc, change))
    if (k != list(d.keys())[-1]): 
      print("———————————————————————————————————————————————————————————————————————")
    else:
      print("=======================================================================")

  print("Total Params: {}".format(sum_parameter(d)))
  print("Trainable Params: {}".format(sum_parameter(d)))
  print("Non-trainable Params: {}".format(sum_parameter(d)))
  print()

d = {1: ["conv2d (Conv2D)", "(None, 123, 123, 32)", 2432],
  2: ["conv2d_1 (Conv2D)", "(None, 121, 121, 32)", 9248],
  3: ["max_pooling2d (MaxPooling2D) ", "(None, 40, 40, 32)", 0]}
  
print_layer(d)
