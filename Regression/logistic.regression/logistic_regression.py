import numpy as np
import pandas as pd
import math


def load_data(file_name):
    """Load dataset from file and return a tuple with a dataset with independent variables (An),
    dataset with dependent variables (y), number of variables"""

    # Get dataset from file using pandas functions
    dataframe = pd.read_csv(file_name, sep=',', header=None)

    # Get number of independent variables input
    x_count = np.size(dataframe, 1) - 1

    # Separate independent data from dataset
    x = dataframe[dataframe.columns[0:x_count]]

    # Separate dependent data from dataset
    y = dataframe[dataframe.columns[-1]]
    return (x, y, x_count, len(y))


def boundary_function(intersection, slopes, features):
    return intersection + sum(slope * feature for slope, feature in zip(slopes, features))


def sigmoid_function(boundary):
    return 1 / (1 + math.exp(-boundary))


def cost_function(subset, intersection, slopes, x, y, samples_number):
    cost = 0
    for features_line, label in zip(x.iterrows(), y):
        boundary = boundary_function(intersection, slopes, features_line[1])
        hypothesis = sigmoid_function(boundary)
        if subset[label] == 1:
            cost += math.log(hypothesis)
        else:
            cost += (1 - math.log(hypothesis))
    cost = -cost / samples_number
    return cost


def gradient_descend(subset, intersection, slopes, x, y, features_number):

    # Gradient of intersection (b0)
    inter_gradient = 0

    # Keep the gradient calculated in each slope
    gradient = [0] * features_number

    for features_line, label in zip(x.iterrows(), y):
        boundary = boundary_function(intersection, slopes, features_line[1])
        hypothesis = sigmoid_function(boundary)
        label_value = subset[label]
        inter_gradient += (hypothesis - label_value)
        for index, feature in zip(range(0, features_number), features_line[1]):
            gradient[index] += ((hypothesis - label_value) * feature)

    return (inter_gradient, gradient)


def logistic_regression(set, x, y, features_number, samples_number, learning_rate):
    subset = set[0]
    intersection = set[1][0]
    slopes = set[1][1]

    intersection_delta, delta = gradient_descend(subset, intersection, slopes, x, y, features_number)

    intersection = intersection - ((learning_rate * intersection_delta) / samples_number)
    for index in range(0, features_number):
        slopes[index] = slopes[index] - ((learning_rate * delta[index]) / samples_number)

    print("Cost: ", cost_function(subset, intersection, slopes, x, y, samples_number))
    intersection_delta = [intersection_delta] + slopes
    labels = {"Gradients": intersection_delta}
    print(pd.DataFrame.from_records(labels))
    return [intersection, slopes]


def logistic_machine(file_name):
    # Load dataset
    x, y, features_number, samples_number = load_data(file_name)
    learning_rate = 0.01

    # Create a set with the possible output and initial slopes and intersection
    subset = {}
    for label in y:
        subset[label] = 0

    label_set = {}
    for label in subset:
        label_set[label] = [subset.copy(), [1, [1] * features_number]]
        label_set[label][0][label] = 1

    repeat = 1
    while repeat != 0:
        repeat = int(input("(0) Stope Machine (Others Numbers) Machine Interation: "))
        for steps in range(0, repeat):
            print("\nONE-VS-ALL:")
            for label in subset:
                print(label, end='-')
                label_set[label][1] = logistic_regression(label_set[label], x, y, features_number, samples_number, learning_rate)


logistic_machine("irisdata.txt")
