import numpy as np
import pymc as pm
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
plt.style.use('ggplot')

count_data = np.loadtxt("BookSource/Chapter1_Introduction/data/txtdata.csv")

plt.hist(count_data, bins = 20)
plt.xlabel('text messages per day')
plt.ylabel('frequency')
plt.show()