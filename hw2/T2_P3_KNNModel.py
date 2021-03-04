import numpy as np
from scipy import stats

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __starDist(self, x1, x2):
        return ((x1[0] - x2[0])/3)**2 + (x1[1] - x2[1])**2

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.

        preds = []
        for i in range(len(X_pred)):
            L = []
            for j in range(len(self.X)):

                if i == j:
                    continue

                L.append((j, self.__starDist(X_pred[i], self.X[j])))

            L.sort(key = lambda x : x[1])

            indexes = [x[0] for x in L[:self.K]]

            pts = [self.y[index] for index in indexes]

            preds.append(stats.mode(pts)[0][0])



        return np.array(preds)



    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y