import numpy as np
import pymc as pm
from ch1 import count_data
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.hist(count_data, bins = 20)
plt.xlabel('text messages per day')
plt.ylabel('frequency')
plt.show()