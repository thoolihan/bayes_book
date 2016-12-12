import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

r_data = np.array([2.725, 2.830, 2.987, 2.981, 3.151, 3.154, 3.088, 2.990])

mu = pm.Uniform('mu', lower=1.5, upper=3.5)

@pm.deterministic
def mu_(mu = mu):
    return np.array([mu] * r_data.size)

observation = pm.Normal("obs", mu = mu_, tau = 1./100, value=r_data, observed=True)
model = pm.Model ([observation, mu])

mcmc = pm.MCMC(model)
mcmc.sample(5e4, 1e3)

means = mcmc.trace('mu')[:]

plt.hist(means,
         histtype='stepfilled',
         bins=25,
         normed=True,
         label='Posterior Mean',
         alpha = 0.8)
plt.show()