import os
import sys
sys.path.insert(1, '../')
from datetime import datetime

import pandas as pd
import numpy as np
import gym
import gym_keygrid

from rl_research.algorithms import agents, train

folder_name = f"data/exp{datetime.now().strftime('%Y_%m_%d_%H%M')}"

agent_names = agents.keys()

scores = {agent_name: {} for agent_name in agent_names}
task_solved = {agent_name: [] for agent_name in agent_names}

grid_lengths = [30]
num_eps = [4000]
num_iter = 20

for i, (num_ep, grid_length) in enumerate(zip(num_eps, grid_lengths)):
    env = gym.make('keygrid-v0', grid_length=grid_length)
    env.render()
    for it in range(num_iter):
        print(f'iteration {it}')

        for agent_name in agent_names:
            agent = train(env, agents[agent_name], num_ep)
            key = (grid_length, it)
            scores[agent_name][key] = agent.scores
            task_solved[agent_name].append(agent.env_solved)
            print(agent_name, agent.env_solved, np.sum(agent.scores))

os.mkdir(folder_name)

for i, algo in enumerate(agent_names):
    df = pd.DataFrame.from_dict(scores[algo], orient='index').T
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_csv(f'{folder_name}/{algo}.csv', index=False)


df = pd.DataFrame.from_dict(task_solved, orient='columns')
df.to_csv(f'{folder_name}/results.csv', index=False)
