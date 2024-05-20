import numpy as np
from numpy.linalg import norm

def sphere(numRandomPoints, numDimensions, randomNumberGenerator):
    '''
    n independent random points on the surface of d-dim sphere
    :return: n-by-d matrix
    '''
    # Use normal() function to create a random number generator 
    # Generate random numbers following a normal distribution
    D = randomNumberGenerator.normal(size=(numRandomPoints, numDimensions))
    # norm(D, axis) can get two-norm of D along one axis
    # D / norm(D, axis=1) is to normalize each point in the matrix D by its norm 
    # reshape() function is to reflect normalized points from (n, ) to (n, 1)
    return D / norm(D, axis=1)[:, None]

