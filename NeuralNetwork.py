import numpy


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.hidden_nodes = hidden_nodes
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        self.wih = numpy.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
