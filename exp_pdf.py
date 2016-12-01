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

def cdf(actual, lambda_):
    '''
    Calculate poisson probability density function.
    :param actual: z
    :param lambda_: lambda, the expected value
    :return: probability
    '''
    return 1 - np.exp(-1 * lambda_ * actual)


if __name__ == "__main__":
    actual = 20.
    mean = 5.
    den = pdf(actual, 1. / mean)
    cum_den = cdf(actual, 1. / mean)
    print("PDF for {0} with mean of {1} is {2:4.2f}".format(actual, mean, den))
    print("CDF for {0} with mean of {1} is {2:4.2f}".format(actual, mean, cum_den))

