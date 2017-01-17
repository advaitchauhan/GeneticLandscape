"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('bar.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    dat, dat2, the_bars = pickle.load(f)

n_groups = 10

means_men = dat


means_women = dat2


fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, means_men, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='Positive Interactions')

rects2 = plt.bar(index + bar_width, means_women, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='Negative Interactions')

plt.xlabel('Number of Interactions')
plt.ylabel('Number of Genes')
plt.xticks(index + bar_width / 2, the_bars)
plt.legend()

plt.tight_layout()
plt.show()