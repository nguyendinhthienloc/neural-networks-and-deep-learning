"""Run a tiny neural-network demo without the MNIST data set.

The toy task is OR classification:
  class 0: [0, 0]
  class 1: [0, 1], [1, 0], [1, 1]

This keeps the shapes small enough to inspect while still exercising
feedforward, backpropagation, mini-batches, and SGD.
"""

import random

import numpy as np

import network2


def log_step(message):
    print("[mock_demo] {}".format(message))


def one_hot(index, size=2):
    log_step("one_hot: convert class {} into a target vector".format(index))
    result = np.zeros((size, 1))
    result[index] = 1.0
    return result


def build_training_data():
    log_step("build_training_data: create the toy OR examples")
    raw_points = [
        ([0.0, 0.0], 0),
        ([0.0, 1.0], 1),
        ([1.0, 0.0], 1),
        ([1.0, 1.0], 1),
    ]
    return [
        (np.array(point).reshape(2, 1), one_hot(label))
        for point, label in raw_points
    ]


def print_predictions(net, training_data, title):
    log_step("print_predictions: {}".format(title.lower()))
    print(title)
    for x, y in training_data:
        output = net.feedforward(x)
        predicted = int(np.argmax(output))
        expected = int(np.argmax(y))
        point = x.ravel().astype(int).tolist()
        rounded = np.round(output.ravel(), 3).tolist()
        print(
            "  input={} expected={} predicted={} raw_output={}".format(
                point, expected, predicted, rounded))


def main():
    log_step("main: seed random number generators")
    random.seed(7)
    np.random.seed(7)

    log_step("main: build training and reference data")
    training_data = build_training_data()
    reference_data = build_training_data()
    log_step("main: create network")
    net = network2.Network([2, 3, 2], pipeline_logging=True)

    log_step("main: run predictions before training")
    print_predictions(net, reference_data, "Before training")
    log_step("main: train with SGD")
    net.SGD(
        training_data,
        epochs=60,
        mini_batch_size=4,
        eta=1.0,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
    log_step("main: run predictions after training")
    print_predictions(net, reference_data, "After training")


if __name__ == "__main__":
    main()
