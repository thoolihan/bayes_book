import numpy as np
import pymc as pm
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt

colors = ["#348ABD", "#A60628", "#7A68A6", "#467821", "#E24A33"]

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
mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000)

lambda_1_samples = mcmc.trace('lambda_1')[:]
lambda_2_samples = mcmc.trace('lambda_2')[:]
tau_samples = mcmc.trace('tau')[:]

figsize(12.5, 5)

N = tau_samples.shape[0]
expected_texts_per_day = np.zeros(n_count_data)
for day in range(0, n_count_data):
    ix = day < tau_samples
    expected_texts_per_day[day] = (lambda_1_samples[ix].sum() \
                                   + lambda_2_samples[~ix].sum()) / N

plt.plot(range(n_count_data), expected_texts_per_day, lw=4, color = colors[4],
         label = "Expected number of text messages received")
plt.xlim(0, n_count_data)
plt.xlabel("Day")
plt.ylabel("Number of text messages")
plt.title("Number of text messages received versus expected number received")
plt.ylim(0, 60)
plt.bar(np.arange(len(count_data)), count_data, color=colors[0], alpha=0.65,
        label="Observed text messages per day")
plt.legend(loc="upper left")
plt.show()
print(expected_texts_per_day)
