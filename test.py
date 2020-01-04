import numpy

from src.NeuralNetwork import NeuralNetwork

random_matrix = numpy.random.rand(3, 3) - .5
print(random_matrix)

nn = NeuralNetwork(3, 3, 3)

a = numpy.array([1, 2, 3, 4], ndmin=2).T

print(a)