from sequential import *


def test():
    # data = [[np.random.rand() for j in range(10)] for i in range(20)]
    # targets = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    # # data = extract_mnist_images("train-images.idx3-ubyte", 2)
    # # data = convert_grayscale_to_rgb(data) / 255
    
    # # targets = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    # model = Sequential()

    # dense_layer1 = DenseLayer(15, SIGMOID)
    # dense_layer2 = DenseLayer(10, SOFTMAX)

    # model.add_layer(dense_layer1)
    # model.add_layer(dense_layer2)

    # batch_size = 3
    # rate = 0.5
    # model.train_model(data, targets, batch_size, rate)
    inputs = extract_mnist_images("train-images.idx3-ubyte", 2)
    inputs = convert_grayscale_to_rgb(inputs) / 255

    convoLayer = ConvolutionLayer((3, 2), RELU, MAX, 2, 0.01)
    denseLayer = DenseLayer(10, SOFTMAX)

    model = Sequential()
    model.add_layer(convoLayer)
    model.add_layer(denseLayer)

    image = inputs[0]
    model.add_inputs(image)
    output = model.calculate()

    for o in output:
        print(np.array(o).shape)
        print(o)
        
    model.forward_prop(inputs)

    model.print_summary()
    
    model.save_model("save")
    # model.print_summary()
    
    
    cek = Sequential()
    
    cek.load_model("save")
    
    # cek.forward_prop(inputs)
    
    cek.print_summary()
    
    # cek.train_model(data, targets, batch_size, rate)
    

if __name__ == '__main__':
    # main()
    test()