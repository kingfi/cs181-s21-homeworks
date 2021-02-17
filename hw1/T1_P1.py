import numpy as np
import math

data = [(0., 0., 0.),
        (0., 0.5, 0.),
        (0., 1., 0.),
        (0.5, 0., 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 1., 0.5),
        (1., 0., 1.),
        (1., 0.5, 1.),
        (1., 1., 1.)]

alpha = 10

W1 = alpha * np.array([[1., 0.],
                        [0., 1.]])
W2 = alpha * np.array([[0.1, 0.],
                        [0., 1.]])
W3 = alpha * np.array([[1., 0.],
                        [0., 0.1]])


def compute_loss(W):
    ## TO DO
    ""
    summation = 0
    N = len(data)
    for i in range(0, N):
        numerator = 0
        denominator = 0
        yi = data[i][2]
        jlist = list(range(0, N))
        jlist.remove(i)

        for j in jlist:
            xj1 = data[j][0]
            xi1 = data[i][0]
            xj2 = data[j][1]
            xi2 = data[i][1]
            yj = data[j][2]
            d1 = xj1 - xi1
            d2 = xj2 - xi2
            exponent = (-1) * ((W[0][0] * (d1**2)) + ( 2*(W[0][1]) * d1 * d2 )+ (W[1][1] * (d2**2)))
            numerator =  numerator + (math.exp(exponent) * yj)
            denominator = denominator + math.exp(exponent)
        yhat = numerator / denominator
        summation = summation + ((yi - yhat)**2)

    loss = summation

    return loss


print(compute_loss(W1))
print(compute_loss(W2))
print(compute_loss(W3))