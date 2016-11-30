import numpy as np
from scipy.special import gammaln

def pdf(actual, mean):
    '''
    Calculate poisson probability density function.
    :param actual: z
    :param mean: lambda, the expected value
    :return: probability
    '''
    return mean * np.exp(-1 * mean * actual)

if __name__ == "__main__":
    actual = 2687.5
    predicted = 2787.2
    density = pdf(actual, predicted)
    print("Density for {0} with mean of {1} is {2:05.2f}%".format(actual, predicted, density * 100))

