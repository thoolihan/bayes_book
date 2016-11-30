import numpy as np
from scipy.special import gammaln

def pdf(actual, lambda_):
    '''
    Calculate poisson probability density function.
    :param actual: z
    :param lambda_: lambda, the expected value
    :return: probability
    '''
    return lambda_ * np.exp(-1 * lambda_ * actual)

if __name__ == "__main__":
    actual = 5
    mean = 21.5
    density = pdf(actual, 1. / mean)
    print("Density for {0} with mean of {1} is {2:05.2f}%".format(actual, mean, density * 100))

