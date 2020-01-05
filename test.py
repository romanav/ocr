import matplotlib.pyplot
import numpy

from src.NeuralNetwork import NeuralNetwork

nn = NeuralNetwork(3, 3, 3, 0.3)
response = nn.query([1.0, 0.5, -1.5])

with open('resources/mnist_train_100.csv', 'r') as data_file:
    data_list = data_file.readlines()

all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
