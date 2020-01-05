import numpy
import scipy.special


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.hidden_nodes = hidden_nodes
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        self.wih = numpy.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        self._activation_function = lambda x: scipy.special.expit(x)

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self._activation_function(hidden_inputs)
        final_outputs = numpy.dot(self.who, hidden_outputs)
        return final_outputs

    def train(self, input_list, targets_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self._activation_function(hidden_inputs)
        final_outputs = numpy.dot(self.who, hidden_outputs)
