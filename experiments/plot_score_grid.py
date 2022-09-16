import os
import sys
sys.path.insert(1, '../')

import pandas as pd

from rl_research.utils import plot_scores_grid
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    exp = 'data/' + sys.argv[1]
else:
    exp = 'Missing'

if not os.path.isdir(exp):
    raise FileNotFoundError

data = {}

algorithms = [x[:-4] for x in os.listdir(f'{exp}')]
algorithms.sort(key=lambda x: len(x))

for algorithm in algorithms:
    data[algorithm] = pd.read_csv(
        f'{exp}/{algorithm}.csv', header=[0, 1], skipinitialspace=True)

grid_lengths = sorted(
    list(set(data[algorithms[0]].columns.get_level_values(0))),
    key = lambda x: int(x))
num_runs = len(data[algorithms[0]][grid_lengths[0]].columns)

for grid_length in grid_lengths:
    for algorithm in algorithms:
        plot_scores_grid(data[algorithm], grid_length, num_runs, algorithm)
