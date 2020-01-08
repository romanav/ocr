
import numpy

from NeuralNetwork import NeuralNetwork

input_nodes = 28 * 28
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.2


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


def train(train_path):
    training_data_list = read_file(train_path)

    # data_size = len(training_data_list)
    # cnt = 1
    # print("data_size: " + str(data_size))

    for r in training_data_list:
        # print(str(cnt / data_size * 100) + "%")
        # cnt += 1

        all_values = r.split(',')
        inputs = adopt_image_to_neuro(all_values)
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        nn.train(inputs, targets)


def test(test_file):
    score_card = []
    test_data_list = read_file(test_file)
    for val in test_data_list:
        all_values = val.split(',')
        correct_label = int(all_values[0])

        image = adopt_image_to_neuro(all_values)
        outputs = nn.query(image)
        label = numpy.argmax(outputs)

        score_card.append(1 if label == correct_label else 0)

    return numpy.asarray(score_card)


print("training")
train('resources/mnist_train.csv')
print("testing")
result = test('resources/mnist_test.csv')
print("performance:", result.sum() / result.size)
