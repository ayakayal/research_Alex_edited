import copy
import sys
sys.path.insert(1, '../')

import gym
import gym_keygrid

import matplotlib.pyplot as plt

from rl_research.algorithms.reinforce_count_states import train as train_reinforce_state
from rl_research.algorithms.reinforce_count_seq import train as train_reinforce_seq
from rl_research.utils import plot_intrinsic_scores

algorithm_names = ["reinforce_res_st",
                   "reinforce_res_seq"
                   ]

algorithm_functions = [train_reinforce_state,
                       train_reinforce_seq
                      ]

algorithms = list(zip(algorithm_names, algorithm_functions))

agents = []

# 1D env

env = gym.make('keygrid-v0', grid_length=10)
num_eps = 200
logs = False

for algorithm_name, algorithm_function in algorithms:
    agent = algorithm_function(env, num_eps, logs=logs)
    agents.append(agent)

# 2D env

env = gym.make('keygrid-v1', grid_length=8)
num_eps = 200
logs = False

for algorithm_name, algorithm_function in algorithms:
    agent = algorithm_function(env, num_eps, logs=logs)
    agents.append(agent)

# plot the results

fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharey=False)

for i, a in enumerate(agents):
    ax = axes[i//2][i%2]
    if i < 2:
        ax.set_ylim([0, 10])
    else:
        ax.set_ylim([0, 70])

    plot_intrinsic_scores(a.intrinsic_scores, ax, a.space_visitation)

plt.show()
