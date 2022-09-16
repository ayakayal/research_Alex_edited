from collections import defaultdict, deque
import itertools
import random

import numpy as np
import scipy
import grakel
import networkx as nx
import sknetwork
from IPython.display import display, SVG

from .reinforce_baseline import ReinforceBaseline
from ..utils import plot_trajectory

class ReinforceSeqComp(ReinforceBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intrinsic_reward = 0.5
        self.K = 10
        self.start_after = 30
        self.archive_length = 200
        self.sequence_archive = deque(maxlen=self.archive_length)
        self.kernel = grakel.WeisfeilerLehman(normalize=True)

    def play_ep(self, num_ep=1, render=False):
        for n in range(num_ep):
            rewards, actions, states = [], [], []
            state = self.env.reset()
            score = 0
            step = 0
            done = False

            G = nx.DiGraph()
            G.add_node(state)
            G.nodes[state]['label'] = '.'.join(map(str, state))

            probs = self.get_proba()
            while not done and step < self.env._max_episode_steps:
                step += 1
                p = probs[state]
                action_idx = random.choices(range(len(p)), weights=p)[0]
                action = self.env.actions[action_idx]
                states.append(state)
                actions.append(action_idx)

                state, reward, done, _ = self.env.step(action)

                self.state_freq[state] += 1

                if state not in states:
                    G.add_node(state)
                    G.nodes[state]['label'] = '.'.join(map(str, state))
                    G.add_edge(states[-1], state)

                rewards.append(reward)
                score += reward

                if render:
                    env.render()

            graph = list(grakel.graph_from_networkx([G], node_labels_tag='label'))[0]
            bonus = 0
            if graph not in self.sequence_archive:
                grakel_graph = grakel.Graph(graph[0], node_labels=graph[1])
                # display(plot_trajectory(grakel_graph))

                if len(self.sequence_archive) >= self.start_after:
                    self.kernel.fit([grakel_graph])
                    distances = 1 / self.kernel.transform(self.sequence_archive)**2 - 1
                    bonus += len(states) * self.intrinsic_reward * np.mean(np.sort(distances.flatten())[:self.K])
                    # discount = np.array([self.gamma**i for i in range(len(rewards))][::-1])
                    # bonus *= discount
                    rewards[-1] += bonus
                    # print("distance: ", np.sort(distances.flatten())[:self.K])

                self.sequence_archive.append(graph)

            self.add_trajectory(states, actions, rewards)

            self.comp_gain()
            self.score = score
            # self.intrinsic_score = np.sum(bonus)
            self.intrinsic_score = bonus

            # print('intrinsic rewards: ', self.intrinsic_score)
            # print('number of sequences: ', len(self.sequence_archive))
