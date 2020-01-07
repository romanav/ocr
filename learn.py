import numpy

from NeuralNetwork import NeuralNetwork

input_nodes = 28*28
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

nn = NeuralNetwork(
    input_nodes=input_nodes,
    hidden_nodes=hidden_nodes,
    output_nodes=output_nodes,
    learning_rate=learning_rate
)

with open('resources/mnist_train_100.csv', 'r') as data_file:
    training_data_list = data_file.readlines()

for r in training_data_list:
    all_values = r.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    nn.train(inputs, targets)


