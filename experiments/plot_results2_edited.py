import os
import sys
sys.path.insert(1, '../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from rl_research.algorithms import agents

agent_names = list(agents.keys())

if len(sys.argv) > 1:
    exp = 'data/' + sys.argv[1]
else:
    exp = 'Missing'

if not os.path.isdir(exp):
    raise FileNotFoundError

width = 0.2
fig, ax = plt.subplots(figsize=(12, 5))

if len(agent_names) == 4:
    pos = [-1.5, -0.5, 0.5, 1.5]
if len(agent_names) == 3:
    pos = [-1, 0, 1]


results = pd.read_csv(f'{exp}/results.csv', header=[0], skipinitialspace=True)

num_runs = len(results[agent_names[0]].values)

for i, agent in enumerate(agent_names):
    data = results[agent].values

    mean = np.mean(data, axis=0)
    error = np.std(data, axis=0)
    rect = ax.bar(pos[i] * width, mean, width * 0.8,
                  yerr=error, ecolor='black', label=agent)
    ax.bar_label(rect, padding=3)

ax.set_ylabel('% of agent that solved the task')
ax.set_title(f'average of number of task solved over {num_runs} runs')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)

ax.legend()

fig.tight_layout()

plt.show()
