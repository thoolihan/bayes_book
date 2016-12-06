from matplotlib import pyplot as plt
import numpy as np
from ch1 import count_data

plt.style.use('ggplot')

colors = ["#348ABD", "#A60628"]

n_count_data = len(count_data)

plt.bar(np.arange(n_count_data), count_data, color = colors[0])


plt.xlabel("Time (days)")
plt.ylabel("Text messages received")
plt.title("Did user's habits change")
plt.show()
