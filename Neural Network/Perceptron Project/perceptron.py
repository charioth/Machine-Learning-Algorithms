import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


class Perceptron():
    def __init__(self, weights_list, dtype=float, activation_function=None, learning_rate=0.01):
        self.weights = np.array(weights_list)
        self.activation_function = activation_function
        self.dtype = dtype
        self.learning_rate = learning_rate

    @classmethod
    def fromRandomWeights(cls, number_of_weights=1, dtype=float, activation_function=None, learning_rate=0.01):
        weights = np.linspace(0, 100, num=number_of_weights)
        return Perceptron(weights_list=weights, dtype=dtype,
                          activation_function=activation_function, learning_rate=0.001)

    def __default_activation(self, entry):
        """Default activation receive a int value and
        return True if bigger than zero else it returns false"""
        if entry >= 0:
            return 1
        return -1

    def __activation(self, entry):
        if self.activation_function is None:
            return self.__default_activation(entry)
        return self.activation_function

    def guess(self, inputs):
        try:
            if not isinstance(inputs.dtype, np.dtype):
                raise TypeError(
                    "Expected a numpy.ndarray but received a " +
                    str(type(inputs)))
            elif inputs.size != self.weights.size:
                raise ValueError(
                    "inputs size does not match Perceptron weight size")
            total = inputs*self.weights

            return self.__activation(total.sum())

        except TypeError:
            raise
        except ValueError:
            raise

    def bguess(self, inputs):
        return np.array([self.guess(row) for row in inputs])

    def train(self, entry, expected):
        guess = self.guess(entry)
        error = expected - guess

        self.weights = self.weights + entry*error*self.learning_rate

    def batch_train(self, inputs, expecteds):
        for entry, expected in zip(inputs, expecteds):
            self.train(entry, expected)

#--------------------Tests Below--------------------------


def TestInputsWithXYCoordinates():
    """Initializes a test with points in range of 0 to 10"""
    inputs = np.random.randint(10, size=(1000, 2))
    expecteds = []
    for row in inputs:
        if row[1] >= row[0]:
            expecteds.append(1)
        else:
            expecteds.append(-1)
    return (inputs, expecteds)


if __name__ == "__main__":
    # Init the random inputs and brain weights
    inputs, expecteds = TestInputsWithXYCoordinates()
    brain = Perceptron.fromRandomWeights(number_of_weights=2)

    # Creates a graphic to plot the values
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # Pass the points to the brain to see how many it got right
    guesses = brain.bguess(inputs)

    # Uses the array to separate the right guesses from the wrong guesses
    right = inputs[guesses == expecteds]
    wrong = inputs[guesses != expecteds]

    # Train the model using all points as batch (This will be upgraded to use only part in future)
    brain.batch_train(inputs, expecteds)

    # add as scatter points in graph to animate
    right_guesses = ax1.scatter(
        right[:, 0], right[:, 1], c='b', label='guessed right')
    wrong_guesses = ax1.scatter(
        wrong[:, 0], wrong[:, 1], c='r', label='guessed wrong')

    # Define a title
    plt.title("Perceptron - Training")

    # Method called in the animation at each frame so that updates the point colors
    def update(i):
        """Used to animate the test, it updates the points and keep training"""
        if(i != 0):
            # Pass the points to the brain to see how many it got right
            guesses = brain.bguess(inputs)

            # Uses the array to separate the right guesses from the wrong guesses
            right = inputs[guesses == expecteds]
            wrong = inputs[guesses != expecteds]

            # Update the values of the points in each scatter plot
            right_guesses.set_offsets(right)
            wrong_guesses.set_offsets(wrong)

            # Train the model using all points as batch
            brain.batch_train(inputs, expecteds)
        return right_guesses, wrong_guesses

    def main():
        ani = animation.FuncAnimation(
            fig, update, interval=1000, blit=True, frames=10)
        plt.legend(loc="right", bbox_to_anchor=(1, -0.1))
        plt.show()

if __name__ == "__main__":
    main()
