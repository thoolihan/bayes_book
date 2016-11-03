import numpy as np
import pymc as pm

count_data = np.loadtxt("BookSource/Chapter1_Introduction/data/txtdata.csv")
n_count_data = len(count_data)

alpha = 1.0 / count_data.mean()

lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)

tau = pm.DiscreteUniform("tau", lower = 0, upper = n_count_data)

@pm.deterministic
def lambda_(tau = tau, lambda_1 = lambda_1, lambda_2 = lambda_2):
    out = np.zeros(n_count_data)
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    return out

observation = pm.Poisson("obs", lambda_, value= count_data, observed = True)

model = pm.Model([observation, lambda_1, lambda_2, tau])
