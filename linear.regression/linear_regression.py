import pandas as pd
import numpy as np
import decimal as dc


def mean(data):
    """Return the mean of data"""
    return sum(data) / len(data)


def calculate_cost(errors):
    return sum(error * error for error in errors) / len(errors)


def load_data(file_name):
    """Load dataset from file and return a tuple with a dataset with independent variables (An),
    dataset with dependent variables (y), number of variables"""

    # Get dataset from file using pandas functions
    dataframe = pd.read_csv(file_name, header=None, delimiter=r"\s+")

    # Separate index columns from dataset
    dataframe = dataframe.ix[:, 1:np.size(dataframe, 1)]

    # Get number of independent variables input
    x_count = np.size(dataframe, 1) - 1

    # Separate independent data from dataset
    x = dataframe[dataframe.columns[0:x_count]]

    # Separate dependent data from dataset
    y = dataframe[dataframe.columns[-1]]
    return (x, y, x_count, len(y))


def linear_estimation(intersection, slopes, variables_input):
    return (intersection + sum(b * x for b, x in zip(slopes, variables_input)))


def print_linear_equation(intersection, slopes):
    index = 0
    print("Y = " + str(intersection) + "*A" + str(index), end='')
    for slope in slopes:
        index += 1
        print(" + " + str(slope) + "*A" + str(index), end='')
    print(end='\n\n')


def calculate_errors(intersection, slopes, input_table, rotules_output):
    return [linear_estimation(intersection, slopes, variables_input[1]) - y for y, variables_input in zip(rotules_output, input_table.iterrows())]


def calculate_regression_line(x, y, variables_number):
    # mean of each input of the data column
    mean_x = [mean(x[column + 1]) for column in range(0, variables_number)]

    # mean of the independant data
    mean_y = mean(y)

    # for each x do (Xij - mean(Xj))
    xx = []
    for index in range(0, variables_number):
        xx.append([element - mean_x[index] for element in x[index + 1]])

    # for each y do Y - mean(Y)
    yy = [element - mean_y for element in y]

    slopes = []
    # Estimate Bn values
    for elem_xx in xx:
        upper = sum(a * b for a, b in zip(elem_xx, yy))  # somatory of (Xij - mean(Xj))*(Y - mean(Y))
        lower = sum(a * a for a in elem_xx)  # somatory of (Y - mean(Y))
        slopes.append(upper / lower)

    # Calculate linear approximation to b0
    intersection = mean_y - sum(slope * elem for slope, elem in zip(slopes, mean_x))

    return (intersection, slopes)


def estocastic_gradient(x, y, variables_number, sample_number, intersection, slopes, learning_rate):
    repeat = 1
    delta = 1
    while((repeat != 0) and (float(abs(delta)) > 0.00001)):
        for row, output in zip(x.iterrows(), y):
            repeat = int(input("(0) Stope Machine (Others Numbers) Machine Interation: "))
            delta = dc.Decimal(learning_rate * (linear_estimation(intersection, slopes, row[1]) - output)) / sample_number
            if float(abs(delta)) < 0.00001:
                break
            for index, a in zip(range(0, variables_number), row[1]):
                slopes[index] -= float(delta) * a
            intersection -= float(delta)
            cost = (sum(error * error for error in calculate_errors(intersection, slopes, x, y))) / (2 * sample_number)
            print("Gradient: ", delta)
            print("Cost: ", cost)
    print("Gradient: ", delta)
    cost = (sum(error * error for error in calculate_errors(intersection, slopes, x, y))) / (2 * sample_number)
    print("Cost: ", cost)


def gradient_descend(x, y, variables_number, sample_number, intersection, slopes, learning_rate):

    # Keep the gradient calculated in each slope
    gradient = [0] * variables_number

    # Gradient of intersection (b0)
    inter_gradient = 0

    # For every input line
    for x_row, output in zip(x.iterrows(), y):
        # Calculate the error (Estimation of Y - Y)
        estimate = intersection + sum(b * elem for b, elem in zip(slopes, x_row[1]))
        error = linear_estimation(intersection, slopes, x_row[1]) - output

        # b0 = sum(error of each column)
        inter_gradient += error

        # Gradient (step) is equal to the sum of error(i) * Xij
        for index, a in zip(range(0, variables_number), x_row[1]):
            gradient[index] += error * a

    for index in range(0, variables_number):
        slopes[index] = slopes[index] - ((learning_rate * gradient[index]) / (10 * sample_number))

    for g in gradient:
        print("Gradient: ", g)


def linear_regression(dataset_file):
    # Load Dataset
    x, y, variables_number, sample_number = load_data(dataset_file)

    learning_rate = 0.000001
    # Keep approximating
    intersection = 0
    slopes = [0] * variables_number

    repeat = 1
    while repeat != 0:
        repeat = int(input("(0) Stope Machine (Others Numbers) Machine Interation: "))
        for steps in range(0, repeat):
            gradient_descend(x, y, variables_number, sample_number, intersection, slopes, learning_rate)
            cost = calculate_cost(calculate_errors(intersection, slopes, x, y))
            print("Cost: ", cost)


linear_regression("x01.txt")
