# %%
%matplotlib inline
%load_ext autoreload
%autoreload 2

import math
import copy

import numpy as np
import gym                                          
import gym_keygrid
import pandas as pd

import matplotlib.pyplot as plt

# %%
%cd '/home/rmapkay/research_project'
!pip install -e ./gym-envs/gym-keygrid

# %%
import gym_keygrid

# %%
!pip install matplotlib

# %%
!pip install pandas

# %%
!pip install gym

# %%
import sys
sys.path.insert(1, "../")

# %%
from rl_research.utils import plot_policy, plot_states, plot_qtables, plot_state_freq, plot_scores

# %%
pip install --upgrade pip


# %%

python -m venv --upgrade ./venv

# %%
!python -m venv --upgrade ./Alex_env2

# %%
!python --version

# %%
!pip install seaborn

# %%
pip install networkx

# %%
pip install grakel

# %%
pip install tensorflow==2.4.0

# %%
pip install tensorflow-probability==0.12.2

# %%
!pip install scipy

# %% [markdown]
# ### Environment and Optimal Value Function

# %%
from rl_research.algorithms.temporal_difference import td_learning, q_learning

# %%
cd ..

# %%
env = gym.make('keygrid-v0', grid_length=5)

# %%
plot_policy(env, env.opt_policy, title="Optimal Policy")

# %% [markdown]
# ### TD Learning using optimal policy

# %%
v_array = td_learning(env.opt_policy, env, display=False)
_ = plot_states(v_array, annot=v_array, title="Value function")

# %% [markdown]
# ### Q Learning 

# %%
q_table = q_learning(env, display=False, num_episodes=50000)
q_policy = {x: env.actions[np.argmax(q_table[x])] for x in env.states}
_ = plot_policy(env, q_policy, "Q-Learning Policy")

# %% [markdown]
# ### Genetic Algorithm

# %%
from rl_research.algorithms.genetic_algo import train, get_best_agent

# %%
best_agents = train(env, 400)

# %%
policy  = {x: get_best_agent(best_agents).get_action(x) for x in env.states}
plot_policy(env, policy, "GA")

# %% [markdown]
# ### ES Algorithm

# %%
from rl_research.algorithms.env_strat import train

# %%
agent = train(env)

# %%
agent.get_policy()
agent.policy
plot_policy(env, agent.policy, "ES")

# %% [markdown]
# ### REINFORCE

# %%
from rl_research.algorithms.reinforce_baseline import ReinforceBaseline
from rl_research.algorithms.reinforce_count_states import ReinforceCountState
from rl_research.algorithms.reinforce_count_states_actions import ReinforceCountStateAction
from rl_research.algorithms.reinforce_count_seq import ReinforceCountSeq
from rl_research.algorithms.training import train

L = 150
logs = True

# %%
cd ..

# %%
agent_reinforce=train(env,ReinforceBaseline,L,logs,0.5)

# %%
plot_scores(agent_reinforce.scores)
agent_reinforce.get_policy()
plot_policy(env, agent_reinforce.pi, "REINFORCE")
plot_state_freq(agent_reinforce.state_freq)
agent_reinforce.get_proba()

# %%
agent_state=train(env,ReinforceCountState,L,logs,0.9)
plot_scores(agent_state.scores)
agent_state.get_policy()
plot_policy(env, agent_state.pi, "REINFORCE")
plot_state_freq(agent_state.state_freq)
agent_state.get_proba()

# %%
agent_state_action=train(env,ReinforceCountStateAction,L,logs,0.5)
plot_scores(agent_state_action.scores)
agent_state_action.get_policy()
plot_policy(env, agent_state_action.pi, "REINFORCE")
plot_state_freq(agent_state_action.state_freq)
agent_state_action.get_proba()



# %%
agent_seq=train(env,ReinforceCountSeq,L,logs,0.8)
plot_scores(agent_seq.scores)
agent_seq.get_policy()
plot_policy(env, agent_seq.pi, "REINFORCE")
plot_state_freq(agent_seq.state_freq)
agent_seq.get_proba()

# %%
agent= ReinforceBaseline(env)


# %%
agent.env

# %%
agent.train(200)

# %%
agent.train(150)

# %%
plot_scores(agent.scores)
agent.get_policy()
plot_policy(env, agent.pi, "REINFORCE")
plot_state_freq(agent.state_freq)
agent.get_proba()

# %%


# %% [markdown]
# ### Reinforce count state exploration

# %%
from rl_research.algorithms.reinforce_count_states import train

# %%
agent_state_count = train(env,ReinforceCountState, 150, logs=logs)

# %%
plot_scores(agent_state_count.scores)
agent_state_count.get_policy()
plot_policy(env, agent_state_count.pi, "REINFORCE States Count")
plot_state_freq(agent_state_count.state_freq)
agent_state_count.get_proba()

# %% [markdown]
# ### Reinforce count seq exploration

# %%
from rl_research.algorithms.reinforce_count_seq import train

# %%
agent = train(env, L, logs=logs)

# %%
plot_scores(agent.scores, window_size=100)
agent.get_policy()
plot_policy(env, agent.pi, "REINFORCE Seq Count")
plot_state_freq(agent.state_freq)
agent.get_proba()
agent.get_space_visitation()

# %% [markdown]
# ### Reinforce count staes + seq exploration

# %%
from rl_research.algorithms.reinforce_count_seq_states import train

# %%
agent = train(env, L, logs=logs)

# %%
plot_scores(agent.scores)
agent.get_policy()
plot_policy(env, agent.pi, "REINFORCE Seq Count")
plot_state_freq(agent.state_freq)
agent.get_proba()

# %%


# %%


# %%


# %% [markdown]
# ### NS ES

# %%
# from NS_ES import train

# %%
# agent = train(env)

# %% [markdown]
# ### QD ES

# %%
# npop = 40
# alpha = 0.1
# sigma = 1
# generation = 100

# K = 3
# archive = set()
# while len(archive) != K:
#     archive.add(QD_Agent().get_bc(env))

# agent = QD_Agent()


# print("\n"+"*"*100)
# print("TRAINING START\n")

# for gen in range(generation):
#     new_agents, N = create_pop(agent, npop=npop, sigma=sigma)
#     F = np.zeros(npop)
#     D = []
    
#     for i, n_a in enumerate(new_agents):
#         F[i] = n_a.evaluate(env)
#         D.append(n_a.get_bc(env)) 
    
# #     print(R)
    
#     nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(np.array(list(archive)))
#     D = np.mean(nbrs.kneighbors(np.array(R))[0], axis=1)

#     R = normalize_array(F) + normalize_array(D)
    
#     if np.std(R) == 0:
#         A = np.zeros(npop)
#     else:
#         A = (R - np.mean(R)) / np.std(R)
        
#     for key in agent.weights:
#         agent.weights[key] += alpha/(npop*sigma) * (N[key] @ A)
    
#     archive.add(agent.get_bc(env))
#     print(archive)
# #     print(agent.weights['w1'][0])
    
# #     print("Generation:", gen, "Score:", print(F))

# print("\n"+"*"*100)
# print("TRAINING ENDED\n")

# %%


# %%


# %%


# %%
res_r = pd.read_csv('data/reinforce_res.csv')
res_r_seq = pd.read_csv('data/reinforce_res_seq.csv')
res_r_st = pd.read_csv('data/reinforce_res_st.csv')


# %%
np.sum(res_r_st > 49)>0

# %%
np.sum(res_r_seq > 49)>0

# %%


