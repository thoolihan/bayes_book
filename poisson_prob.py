import numpy as np
from scipy.special import gammaln

def _poisson_probability(actual, mean):
    '''
    Calculate poisson probability mass function.
    Since normal implementation is unstables (overflows), uses numerically stable implementation from
    https://en.wikipedia.org/wiki/Poisson_distribution#Definition
    :param actual: k
    :param mean: lambda, the expected value
    :return: probability
    '''
    return np.exp(actual * np.log(mean) - mean - gammaln(actual + 1))

poisson_probability = np.vectorize(_poisson_probability)

if __name__ == "__main__":
    kids = 6
    mean_female = 3
    outcomes = np.arange(kids + 1)
    prob_dist = poisson_probability(outcomes, mean_female)
    for n, p in zip(outcomes, prob_dist):
        print("{n} female(s) {p:7.4f}%".format(n=n, p=100.*p))
