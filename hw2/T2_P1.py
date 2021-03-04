import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import matplotlib.patches as mpatches
import math
from scipy.special import expit as sigmoid

# This script requires the above packages to be installed.
# Please implement the basis2, basis3, fit, and predict methods.
# Then, create the three plots. An example has been included below, although
# the models will look funny until fit() and predict() are implemented!

# You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

# Note: this is in Python 3

def exp(arr, power):
    return [x**power for x in arr]

def basis1(x):
    return np.stack([np.ones(len(x)), x, ], axis=1)


# TODO: Implement this
def basis2(x):
    return np.stack([np.ones(len(x)), x, exp(x, 2), exp(x, 3)], axis=1)


# TODO: Implement this
def basis3(x):
    return np.stack([np.ones(len(x)), x, exp(x, 2), exp(x, 3),
                    exp(x, 4), exp(x, 5)], axis=1)

class LogisticRegressor:
    def __init__(self, eta, runs):
        # Your code here: initialize other variables here
        self.eta = eta
        self.runs = runs

    # NOTE: Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __grad(self, x, y, w):
        x = x.reshape((x.shape[0], 1))
        yhat = [sigmoid(z) for z in w.T @ x]
        return (yhat - y)[0][0] * x


    # TODO: Optimize w using gradient descent
    def fit(self, x, y, w_init=None):


        N = x.shape[0]
        D = x.shape[1]
        if w_init is not None:
            self.W = w_init
        else:
            self.W = np.random.rand(D, 1)
        expected_W_shape = self.W.shape
        for i in range(self.runs):
            grad = 0
            for j in range(N):
                grad += (self.predict(x[j]) - y[j]) * x[j] / N
            self.W = self.W - np.reshape(self.eta * grad, self.W.shape)
        assert self.W.shape == expected_W_shape

    # TODO: Fix this method!
    def predict(self, x):
        return sigmoid(np.dot(x, self.W))

# Function to visualize prediction lines
# Takes as input last_x, last_y, [list of models], basis function, title
# last_x and last_y should specifically be the dataset that the last model
# in [list of models] was trained on
def visualize_prediction_lines(last_x, last_y, models, basis, title):
    # Plot setup
    green = mpatches.Patch(color='green', label='Ground truth model')
    black = mpatches.Patch(color='black', label='Mean of learned models')
    purple = mpatches.Patch(color='purple', label='Model learned from displayed dataset')
    plt.legend(handles=[green, black, purple], loc='upper right')
    plt.title(title)
    plt.xlabel('X Value')
    plt.ylabel('Y Label')
    plt.axis([-10, 10, -.1, 1.1]) # Plot ranges

    # Plot dataset that last model in models (models[-1]) was trained on
    cmap = c.ListedColormap(['r', 'b'])
    plt.scatter(last_x, last_y, c=last_y, cmap=cmap, linewidths=1, edgecolors='black')
    # Plot models
    X_pred = np.linspace(-10, 10, 1000)
    X_pred_transformed = basis(X_pred)

    ## Ground truth model
    plt.plot(X_pred, sigmoid(np.sin(X_pred)), 'g', linewidth=5)

    ## Individual learned logistic regressor models
    Y_hats = []
    for i in range(len(models)):
        model = models[i]
        Y_hat = model.predict(X_pred_transformed)
        Y_hats.append(Y_hat)
        if i < len(models) - 1:
            plt.plot(X_pred, Y_hat, linewidth=.3)
        else:
            plt.plot(X_pred, Y_hat, 'purple', linewidth=3)

    # Mean / expectation of learned models over all datasets
    plt.plot(X_pred, np.mean(Y_hats, axis=0), 'k', linewidth=5)

    plt.savefig(title + '.png')
    plt.show()

# Function to generate datasets from underlying distribution
def generate_data(dataset_size):
    x, y = [], []
    for _ in range(dataset_size):
        x_i = 20 * np.random.random() - 10
        p_i = sigmoid(np.sin(x_i))
        y_i = np.random.binomial(1, p_i)
        x.append(x_i)
        y.append(y_i)
    return np.array(x), np.array(y).reshape(-1, 1)

if __name__ == "__main__":
    eta = 0.001
    runs = 10000
    N = 20

    # TODO: Make plot for each basis with all 10 models on each plot

    for basis in [basis1, basis2, basis3]:

        all_models = []
        for _ in range(10):
            x, y = generate_data(N)
            x_transformed = basis(x)
            model = LogisticRegressor(eta=eta, runs=runs)
            model.fit(x_transformed, y)
            all_models.append(model)


        # Here x and y contain last dataset:
        visualize_prediction_lines(x, y, all_models, basis, basis.__name__)
