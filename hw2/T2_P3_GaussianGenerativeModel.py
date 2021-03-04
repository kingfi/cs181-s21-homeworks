import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __hot(self, y):
        newY = []
        for i in range(len(y)):
            z = list(np.zeros(len(np.unique(y)) - 1, dtype = int))
            newY.append(list(np.insert(z, y[i], 1)))
        return np.array(newY)

    def __mu(self, X, y):
        mu = []
        N = len(y)
        for k in range(3):
            numerator = np.sum([X[n] if y[n] == k else 0 for n in range(N)])
            denominator = np.sum([1 if y[n] == k else 0 for n in range(N)])
            mu.append(np.divide(numerator, denominator))
        return np.array(mu).T

    def __sigmaSingle(self, X, y, mu, k):
        cov, N = np.zeros((X.shape[1], X.shape[1])), X.shape[0]
        def diff(i): return (X[i] - mu[:,k]).reshape((X.shape[1], 1))
        return sum([diff(i) @ diff(i).T if y[i] == k else 0 for i in range(N)])

    def __pi(self, X, y):
        N = len(y)
        numerator = 0
        pi = []
        for k in range(3):
            numerator = np.sum([1 if y[n] == k else 0 for n in range(N)])
            pi.append(numerator / N)
        pi = np.array(pi)
        return pi.reshape((pi.shape[0], 1))


    # TODO: Implement this method!
    def fit(self, X, y):

        self.mu = self.__mu(X, y)

        if self.is_shared_covariance:
            self.cov = sum([self.__sigmaSingle(X, y, self.mu, k) for k in range(3)]) / X.shape[0]
        else:
            self.cov = [self.__sigmaSingle(X, y, self.mu, k)/ len(X[y==k]) for k in range(3)]

        self.pi = self.__pi(X, y)


        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for x in X_pred:

            if self.is_shared_covariance == True:
                preds.append(np.argmax([self.pi[k]*mvn(self.mu[:,k], self.cov).pdf(x) for k in range(3)]))
            else:
                preds.append(np.argmax([self.pi[k]*mvn(self.mu[:,k], self.cov[k]).pdf(x) for k in range(3)]))

        return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        loss = 0


        if self.is_shared_covariance == True:
            loss = sum([-sum(np.multiply([1 if y[i] == k else 0 for i in range(len(X))],
                                        (np.log(np.multiply(mvn(self.mu[:,k], self.cov).pdf(X), self.pi[k])))  )) for k in range(3)])
        else:
            loss = sum([-sum(np.multiply([1 if y[i] == k else 0 for i in range(len(X))],
                                        (np.log(np.multiply(mvn(self.mu[:,k], self.cov[k]).pdf(X), self.pi[k])))  )) for k in range(3)])

        return loss



