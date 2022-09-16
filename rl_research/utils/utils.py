import random
import numpy as np
import grakel
import networkx as nx

def argmax_tiebreaker(arr):
    return np.random.choice(np.flatnonzero(arr == arr.max()))


def greedy_policy(state, q_array):
    action_idx = argmax_tiebreaker(q_array[state[0], state[1], :])
    return action_idx


def epsilon_greedy_policy(state, q_array, epsilon, actions):
    if np.random.rand() > epsilon:
        action_idx = greedy_policy(state, q_array)
    else:
        action_idx = random.choice(range(len(actions)))
    return action_idx

def trajectory2graph(trajectory):
    G = nx.DiGraph()
    s = trajectory[0]
    G.add_node(s)
    G.nodes[s]['label'] = ''.join(map(str, s))
    for i, s in enumerate(trajectory[1:]):
        if s not in G.nodes:
            G.add_node(s)
            G.nodes[s]['label'] = ''.join(map(str, s))
            G.add_edge(trajectory[i], s)

    graph_info = list(grakel.graph_from_networkx([G], node_labels_tag='label'))[0]
    graph = grakel.Graph(graph_info[0], node_labels=graph_info[1])
    return graph
