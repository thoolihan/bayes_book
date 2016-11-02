from matplotlib import pyplot as plt
import numpy as np

plt.style.use('ggplot')

colors = ["#348ABD", "#A60628"]

prior = [1/21., 20/21.]
posterior = [0.087, 1 - 0.087]

plt.bar([0, .7], prior, alpha=0.7, width = 0.25,
        color = colors[0], label = "prior distribution",
        lw = "3", edgecolor = colors[0])

plt.bar([0 + 0.25, .7 + 0.25], posterior, alpha=0.7, width = 0.25,
        color = colors[1], label = "prior distribution",
        lw = "3", edgecolor = colors[1])

plt.xticks([0.2, 0.95], ["Librarian", "Farmer"])
plt.ylabel("Probability")
plt.title("Prior and posterior probabilites of Steve's Occupation")
plt.legend(loc = "upper left")
plt.show()
