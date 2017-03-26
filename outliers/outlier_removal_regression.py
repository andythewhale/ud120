#!/usr/bin/python

import random
import numpy
import matplotlib.pyplot as plt
import pickle

from outlier_cleaner import outlierCleaner

### load up some practice data with outliers in it
ages = pickle.load( open("practice_outliers_ages.pkl", "r") )
net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "r") )

### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

### fill in a regression here!  Name the regression object reg so that
from sklearn import linear_model
import numpy as np


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    # to calculate error as specified in the code comment
    errors = abs(net_worths - predictions)
    cleaned_data =zip(ages, net_worths, errors)

    # the [0] isn't necessary in this case since errors is a 1-D array
    cleaned_data = sorted(cleaned_data,key=lambda x:x[2], reverse=True)
    limit = int(len(net_worths) * 0.1)

    # cast the iterator object as a list. I needed to do this to avoid errors
    # in the calling code.
    return list(cleaned_data[limit:])

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(ages_train, net_worths_train)

# Coefficients
print('slope: ', reg.coef_)
print('intercept: ', reg.intercept_)

# The mean square error
print("Residual sum of squares: %.2f" % np.mean((reg.predict(ages_test) - net_worths_test) ** 2))
print('score train ' , reg.score(ages_train,net_worths_train))
print('score test ' , reg.score(ages_test,net_worths_test))



### the plotting code below works, and you can see what your regression looks like
try:
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.show()

### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"

### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    try:
        reg.fit(ages , net_worths)
        plt.plot(ages, reg.predict(ages), color="blue")
        print('slope cleaned: ', reg.coef_)
        print('intercept cleaned: ', reg.intercept_)
        print('score train: ' , reg.score(ages_train,net_worths_train))
        print('score test: ' , reg.score(ages_test,net_worths_test))

    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"

    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()


else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"