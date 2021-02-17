#####################
# CS 181, Spring 2021
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt
import math

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40

    if part == "a" and not is_years:
        xx = xx/20

    if part == "a":
        arr = []
        arr = [[ x**j for j in range(6)] for x in xx]
        return np.array(arr)

    if part == 'b':
        arr = []
        arr = [ ([1] + [math.exp(( -(x - j)**2) / 25) for j in range(1960, 2011, 5)]) for x in xx]
        return np.array(arr)

    if part == 'c':
        arr = []
        arr = [ ([1] + [ math.cos(x / j) for j in range(1, 6)]) for x in xx]
        return np.array(arr)

    if part == 'd':
        arr = []
        arr = [ ([1] + [ math.cos(x / j) for j in range(1, 26)]) for x in xx]
        return np.array(arr)

    return

#print("basis ", make_basis(years, part="b", is_years=True))
# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_sunspots = np.linspace(0, 160, 200)
#grid_X = np.vstack((np.ones(grid_years.shape), grid_years))

print("Years Loss: ")
for part in ['a', 'b', 'c', 'd']:
    w = find_weights(make_basis(years, part= part, is_years=True), Y)
    basis = make_basis(grid_years, part=part)
    grid_Yhat  = np.dot(basis, w)
    # TODO: plot and report sum of squared error for each basis

    loss = 0
    for i in range(len(Y)):
        yhat = np.dot(make_basis(years, part = part), w)
        y = Y[i]
        loss = loss + (y - yhat[i])**2
    print('L2: ', loss)

    # Plot the data and the regression line
    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.savefig('Year_' + part + '.png')
    plt.show()

print("Sunspot Loss: ")
for part in ['a', 'c', 'd']:
    w = find_weights(make_basis(sunspot_counts[years < last_year], part= part, is_years=False), Y[years < last_year])
    basis = make_basis(grid_sunspots, part = part, is_years=False)
    grid_Yhat  = np.dot(basis, w)


    loss = 0
    for i in range(len(Y[years < last_year])):
        yhat = np.dot(make_basis(sunspot_counts[years < last_year], part = part, is_years=False), w)
        y = Y[i]
        loss = loss + (y - yhat[i])**2
    print('L2: ', loss)

    # Plot the data and the regression line
    plt.plot(sunspot_counts[years < last_year], republican_counts[years < last_year], 'o', grid_sunspots, grid_Yhat, '-')
    plt.xlabel("Sunspot Count")
    plt.ylabel("Number of Republicans in Congress")
    plt.savefig('Sunspot' + part + '.png')
    plt.show()





