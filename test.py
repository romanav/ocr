from src.NeuralNetwork import NeuralNetwork

nn = NeuralNetwork(3, 3, 3, 0.3)
response = nn.query([1.0, 0.5, -1.5])

print(response)
