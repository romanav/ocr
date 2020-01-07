import numpy

from NeuralNetwork import NeuralNetwork

input_nodes = 28 * 28
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3


def adopt_image_to_neuro(values):
    return (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01


def read_file(file_path):
    with open(file_path, 'r') as data_file:
        return data_file.readlines()


nn = NeuralNetwork(
    input_nodes=input_nodes,
    hidden_nodes=hidden_nodes,
    output_nodes=output_nodes,
    learning_rate=learning_rate
)


def train():
    training_data_list = read_file('resources/mnist_train_100.csv')
    for r in training_data_list:
        all_values = r.split(',')
        inputs = adopt_image_to_neuro(all_values)
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        nn.train(inputs, targets)


def test():
    test_file = 'resources/mnist_test_10.csv'
    test_data_list = read_file(test_file)
    for val in test_data_list:
        all_values = val.split(',')
        print(all_values[0])
        image = adopt_image_to_neuro(all_values)
        print(nn.query(image))
        print()


train()
test()
