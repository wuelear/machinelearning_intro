__author__ = 'xxxh'

"""Softmax."""

#one-dimensional array (which is interpreted as a column vector representing a single sample)
scores0 = [3.0, 1.0, 0.2]
#2-dimensional array where each column represents a sample
scores1 = [[1, 2, 3, 6],
           [2, 4, 5, 6],
           [3, 8, 7, 6]]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    #calculate the exponential of all elements in the array
    #all rows  - all columns
    expon = np.exp(x)
    #calculate the sum of each line in the array
    #0 -> sum of each column
    #1 -> sum of each line
    #If axis is negative it counts from the last to the first axis.
    #-> if there are multiple rows -> sum up each row
    #-> if there is just one row -> sum up each row column
    #-> a feature is defined as the column (not the line!)
    exponSum = np.sum( expon, axis=0 )
    #exponSum is now an array with as many rows as the input array has columns
    #-> it contains the summed up exponential values for each column's elements
    #-> we need to transform it into a column array with as many lines as the input has lines
    exponSumT = np.array( exponSum )
    result = expon / exponSumT

    return result

#print( "Softmax result: " )
probabilities0 = softmax( np.array( scores0 ) )
sum0 = probabilities0.sum()
probabilities1 = softmax( np.array( scores1 ) )
sum1 = probabilities1.sum()
print( probabilities1 )

# Plot softmax curves
import matplotlib.pyplot as plt

# arrange: returns evenly spaced values within a given interval
# start - stop - step
x = np.arange(-2.0, 6.0, 0.1)

#vstack: takes single arrays and put them together
#        -> each array in its own row
#ones_like: returns an array of ones with the same shape as the given one
scores2 = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
#one row for each score -> the output of a logistic classifier
#-> each column represents the score that an input belongs to a specific class
#print( scores2 )
probabilities2 = softmax( scores2 *10 )
sum2 = probabilities2.sum(1)

plt.plot(x, softmax(scores2).T, linewidth=2)
plt.show()
print( 'bla' )
