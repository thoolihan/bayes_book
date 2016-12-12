import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
plt.style.use('ggplot')

tau = pm.rdiscrete_uniform(0, 80)
print("tau: {0:.2f}".format(tau))

alpha = 1./20.
lambda_1, lambda_2 = pm.rexponential(alpha, 2)
print("lambda_1: {0:.2f} lambda_2:{1:.2f}".format(lambda_1, lambda_2))

lambda_ = np.r_[lambda_1 * np.ones(tau), lambda_2 * np.ones(80 - tau)]
print("lambda_: {0}".format(lambda_))

data = pm.rpoisson(lambda_)
print("data: {0}".format(data))

plt.bar(np.arange(80), data)
plt.bar(tau - 1, data[tau-1], color="r", label="user behavior change")
plt.xlabel("Time (days)")
plt.ylabel("Text messages received")
plt.title("Simulation")
plt.xlim(0, 80)
plt.legend()
plt.show()