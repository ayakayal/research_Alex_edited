import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

if len(sys.argv) > 1:
    exp = 'data/' + sys.argv[1]
else:
    exp = 'Missing'

if not os.path.isdir(exp):
    raise FileNotFoundError

algorithms = [x[:-4] for x in os.listdir(f'{exp}') if x!='results.csv']
algorithms.sort(key = lambda x: len(x))

data = {}
for algorithm in algorithms:
    data[algorithm] = pd.read_csv(
        f'{exp}/{algorithm}.csv', header=[0, 1], skipinitialspace=True)

grid_lengths = sorted(
    list(set(data[algorithms[0]].columns.get_level_values(0))),
    key = lambda x: int(x))
num_runs = len(data[algorithms[0]][grid_lengths[0]].columns)

results = {}
for grid_length in grid_lengths:
    for algorithm in algorithms:
        results[(grid_length, algorithm)] = data[str(
            algorithm)][str(grid_length)].sum()

results = pd.DataFrame.from_dict(results, orient='columns')
display(results)

labels = [f'Grid Length of {gl}' for gl in grid_lengths]

x = np.arange(len(labels))
width = 0.2
fig, ax = plt.subplots(figsize=(12, 5))

if len(algorithms) == 4:
    pos = [-1.5, -0.5, 0.5, 1.5]
if len(algorithms) == 3:
    pos = [-1, 0, 1]

for i, algorithm in enumerate(algorithms):
    data = results.xs(algorithm, axis=1, level=1, drop_level=False).values

    mean = np.mean(data, axis=0)
    error = np.std(data, axis=0)
    rect = ax.bar(x + pos[i] * width, mean, width * 0.8,
                  yerr=error, ecolor='black', label=algorithm)
    ax.bar_label(rect, padding=3)

ax.set_ylabel('total rewards')
ax.set_title(f'average of total rewards over {num_runs} runs')
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.legend()

fig.tight_layout()

plt.show()
