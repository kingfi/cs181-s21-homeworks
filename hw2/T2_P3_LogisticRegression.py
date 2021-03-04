import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt


# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.runs = 10000
        self.losses = []

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    """def __grad(self, x, y):
        x = x.reshape((x.shape[0], 1))
        y = y.reshape((y.shape[0], 1))
        yhat = softmax(self.W.T @ x)
        return (((yhat - y)[0] * x)).reshape((x.shape[0]))"""

    def __grad2(self, X, Y, W):
        gradient = np.zeros((X.shape[1], 3))

        for i in range(X.shape[0]):
            soft = softmax(np.dot(X[i], W))
            for j in range(X.shape[1]):
               gradient[:,j] += (soft[j] - Y[i][j]) * X[i]
        return np.add(gradient, self.lam * W)

    def __hot(self, y):
        newY = []
        for i in range(len(y)):
            z = list(np.zeros(len(np.unique(y)) - 1, dtype = int))
            newY.append(list(np.insert(z, y[i], 1)))
        return np.array(newY)

    def fit(self, X, y):

        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)


        self.W = np.random.rand(X.shape[1], 3)

        y = self.__hot(y)

        #Gradient Descent Process
        for run in range(self.runs):

            """gradient = np.zeros((X.shape[1], 3))

            for i in range(0, len(X)):
                gradient = np.add(gradient, self.__grad(X[i], y[i]))"""
            gradient = self.__grad2(X, y, self.W)

            loss = -np.sum(np.add(np.multiply(y, np.log(softmax(X @ self.W))), self.lam * (np.linalg.norm(self.W))**2 ))
            self.losses.append(loss)

            #w becomes initial w - learning rate times grad
            self.W = np.subtract(self.W, (self.eta * gradient))

        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.


        """X_pred = np.insert(X_pred, 0, np.ones(X_pred.shape[0]), axis=1)
        y_pred = np.argmax(softmax(self.W @ X_pred.T), axis = 1)
        return y_pred"""

        preds = []
        for x in X_pred:
            preds.append(np.argmax(softmax(self.W.T @ (np.insert(x, 0, [1])))))
        return np.array(preds)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        plt.plot(self.losses)
        plt.savefig(output_file + '.png')


