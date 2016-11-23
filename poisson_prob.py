import numpy as np

def poisson_probability(actual, mean):
    #overflows: (mean ** candidate * np.e ** (-1 * mean)) / factorial(candidate)
    p = np.e ** -mean
    for i in range(actual):
        p *= mean
        p /= i+1
    return p

def poisson_map(a, mean):
    return [poisson_probability(mean, c) for c in a]

if __name__ == "__main__":
    mean = 2
    candidate = 4
    print("Weekly average: {average}, Probability of {candidate} is {probability:.2f}%".format(
        average = mean, candidate = candidate, probability = poisson_probability(candidate, mean) * 100))
