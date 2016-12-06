import numpy as np
import pymc as pm
from ch1 import count_data

colors = ["#348ABD", "#A60628", "#7A68A6", "#467821", "#E24A33"]

n_count_data = len(count_data)

alpha = 1. / count_data.mean()

lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)

tau = pm.DiscreteUniform("tau", lower = 0, upper = n_count_data)

@pm.deterministic
def lambda_(tau = tau, lambda_1 = lambda_1, lambda_2 = lambda_2):
    out = np.zeros(n_count_data)
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    return out

observation = pm.Poisson("obs", lambda_, value = count_data, observed = True)

model = pm.Model([observation, lambda_1, lambda_2, tau])
mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000)

lambda_1_samples = mcmc.trace('lambda_1')[:]
lambda_2_samples = mcmc.trace('lambda_2')[:]
tau_samples = mcmc.trace('tau')[:]

print("\n\n")
print("lambda 1 is less than lambda 2 {0:.4f}% of the time".format(100 * (lambda_1_samples < lambda_2_samples).mean()))

for d in [1,2,5,10]:
    v = (abs(lambda_1_samples - lambda_2_samples) >= d).mean()
    print("What is the probability the difference is larger than {0}?: {1:.4f}%".format(d, v * 100))

exp_lift = ((lambda_2_samples - lambda_1_samples) / lambda_1_samples).mean()
print("Expected lift: {0:.4f}%".format(exp_lift * 100.))
