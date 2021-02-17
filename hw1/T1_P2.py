#####################
# CS 181, Spring 2021
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Read from file and extract X and y
df = pd.read_csv('data/p2.csv')

X_df = df[['x1', 'x2']]
y_df = df['y']

X = X_df.values
y = y_df.values

print("y is:")
print(y)

W1 = np.array([[1., 0.],
                [0., 1.]])
N = len(df)

def predict_kernel(alpha=0.1):
    """Returns predictions using kernel-based predictor with the specified alpha."""
    # TODO: your code here
    yhats = []
    W = W1 * alpha
    for i in range(0, N):
        numerator = 0
        denominator = 0
        yi = df['y'][i]
        jlist = list(range(0, N))
        jlist.remove(i)

        for j in jlist:
            xj1 = df['x1'][j]
            xi1 = df['x1'][i]
            xj2 = df['x2'][j]
            xi2 = df['x2'][i]
            yj = df['y'][j]
            d1 = xj1 - xi1
            d2 = xj2 - xi2
            exponent = (-1) * ((W[0][0] * (d1**2)) + ( 2*(W[0][1]) * d1 * d2 )+ (W[1][1] * (d2**2)))
            numerator =  numerator + (math.exp(exponent) * yj)
            denominator = denominator + math.exp(exponent)
        yhat = numerator / denominator
        yhats.append(yhat)
    return yhats

def predict_knn(k=1):
    """Returns predictions using KNN predictor with the specified k."""
    # TODO: your code here
    yhats = []
    W = W1
    for i in range(0, N):
        L = []
        jlist = list(range(0, N))
        jlist.remove(i)

        for j in jlist:
            xj1 = df['x1'][j]
            xi1 = df['x1'][i]
            xj2 = df['x2'][j]
            xi2 = df['x2'][i]
            yj = df['y'][j]
            d1 = xj1 - xi1
            d2 = xj2 - xi2
            mdistance = math.sqrt(((W[0][0] * (d1**2)) + ( 2*(W[0][1]) * d1 * d2 )+ (W[1][1] * (d2**2))))
            L.append((j, mdistance))

        L.sort(key = lambda x : x[1])

        indexes = [x[0] for x in L[:k]]

        pts = [df['y'][index] for index in indexes]

        yhats.append(np.mean(pts))

    return yhats



def plot_kernel_preds(alpha):
    title = 'Kernel Predictions with alpha = ' + str(alpha)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_kernel(alpha)
    print(df['x1'], df['x2'], y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2),
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center')

    # Saving the image to a file, and showing it as well
    plt.savefig('alpha' + str(alpha) + '.png')
    plt.show()

def plot_knn_preds(k):
    title = 'KNN Predictions with k = ' + str(k)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_knn(k)
    print(y_pred)
    print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2),
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center')
    # Saving the image to a file, and showing it as well
    plt.savefig('k' + str(k) + '.png')
    plt.show()


for alpha in (0.1, 3, 10):
    # TODO: Print the loss for each chart.
    plot_kernel_preds(alpha)
    yvals = pd.Series(y)
    ypred = pd.Series(predict_kernel(alpha))
    loss =  ((yvals.subtract(ypred)).apply(lambda x : x**2)).sum()
    print("Kernel, alpha = ",alpha, ". Loss is ", loss)


for k in (1, 5, len(X)-1):
    # TODO: Print the loss for each chart.
    plot_knn_preds(k)
    yvals = pd.Series(y)
    ypred = pd.Series(predict_knn(k))
    loss =  ((yvals.subtract(ypred)).apply(lambda x : x**2)).sum()
    print("Kernel, k = ",k, ". Loss is ", loss)
