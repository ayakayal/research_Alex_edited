import random
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")
import sknetwork
from IPython.display import SVG

def plot_policy(env, policy, title):
    policy = np.array([policy[s] for s in env.states])
    policy = policy.reshape((2, -1, env.len), order='F')
    labels = np.vectorize(lambda x: env.actions_dic[x])(policy)
    plot_states(policy, title=title, cbar=False, annot=labels, fmt='s')


def plot_states(state_array, title=None, figsize=(10, 10), annot=True,
                fmt="0.1f", linewidths=.5, square=True, cbar=False, cmap="Reds", ax=None):
    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize)

    states_with_key = state_array[0]
    states_without_key = state_array[1]

    annot_with_key = annot[0]
    annot_without_key = annot[1]

    sns.heatmap(states_with_key, annot=annot_with_key, fmt=fmt, linewidths=linewidths,
                square=square, cbar=cbar, cmap=cmap, ax=axes[0],
                cbar_kws={"orientation": "horizontal"},
                annot_kws={"size": int(figsize[1] * 0.8)}) #it was 1.2
    axes[0].set_title('States without Key')

    sns.heatmap(states_without_key, annot=annot_without_key, fmt=fmt, linewidths=linewidths,
                square=square, cbar=cbar, cmap=cmap, ax=axes[1],
                cbar_kws={"orientation": "horizontal"},
                annot_kws={"size": int(figsize[1] * 0.8)})
    axes[1].set_title('States with Key')

    if title is not None:
        plt.suptitle(title)

    if ax is None:
        plt.show()
    else:
        return plt #it was ax


def plot_qtables(q_array, title=None, figsize=(7, 7), annot=True,
                 fmt="0.1f", linewidths=.5, square=True, cbar=False, cmap="Reds"):
    num_actions = q_array.shape[-1]

    global_figsize = list(figsize)
    global_figsize[0] *= num_actions
    # Sample figsize in inches
    fig, ax_list = plt.subplots(nrows=num_actions, figsize=global_figsize)

    for action_index in range(num_actions):
        ax = ax_list[action_index]
        state_seq = q_array[:, :, action_index].T
        plot_states(state_seq, title=f'Action {action_index}', figsize=figsize, annot=True,
                    fmt="0.1f", linewidths=.5, square=True, cbar=False, cmap="Reds", ax=ax)

    plt.suptitle(title)
    plt.show()


def plot_state_freq(state_freq):
    x = np.arange(len(state_freq))
    y = state_freq.values()
    labels = [str(x) for x in state_freq.keys()]
    plt.figure(figsize=(10, 5))
    plt.bar(x, y, tick_label=labels)
    _ = plt.xticks(rotation='vertical')

#def plot_state_freq_heatmap(state_freq):


# def plot_scores(scores, window_size=100):
#     ma = np.convolve(scores, np.ones(window_size, dtype=int), 'valid')
#     ma /= window_size
#     x = np.arange(len(scores))
#     plt.figure(figsize=(10, 5))
#     plt.figure(figsize=(15, 7))
#     plt.xlabel("episodes")
#     plt.ylabel("rewards")
#     plt.scatter(x, scores, marker='+', c='b', s=30, linewidth=1,
#                 alpha=0.5, label="total rewards")
#     plt.plot(x[window_size - 1:], ma, c='r',
#              alpha=0.7, label="reward moving average")


def plot_scores_grid(data, label, num_runs, a):
    c = 3
    r = int(round(num_runs / c + 0.49, 0))

    fig, axes = plt.subplots(r, c, figsize=(r * 6, c * 5), sharey=True)
    for n in range(num_runs):
        ax = axes[n // c][n % c]
        scores = data[(label, str(n))]
        window_size = 100
        ma = np.convolve(scores, np.ones(window_size, dtype=int), 'valid')
        ma /= window_size
        x = np.arange(len(scores))
        ax.set_xlabel("episodes")
        ax.set_ylabel("rewards")
        #ax.title.set_text(f'run {n}')
        ax.scatter(x, scores, marker='+', c='b', s=30, linewidth=1,
                   alpha=0.5, label="total rewards")
        ax.plot(x[window_size - 1:], ma, c='r',
                alpha=0.7, label="reward moving average")

    for i in range(n + 1, r * c):
        ax = axes[i // c][i % c]
        ax.axis('off')

    fig.suptitle(f'Total rewards over each run for {a}')

    #plt.show()
    return plt

def plot_scores(intrinsic_scores, ax=None, space_visitation=None):
    window_size = 100 #change to 100
    ma = np.convolve(intrinsic_scores, np.ones(window_size, dtype=int), 'valid')
    ma /= window_size

    
    fig, ax = plt.subplots(1, 1)

    x = np.arange(len(intrinsic_scores))

    ax.set_xlabel("episodes")
    ax.set_ylabel("rewards")
    ax.scatter(x, intrinsic_scores, marker='+', c='b', s=30, linewidth=1,
                alpha=0.5, label="total rewards")
    ax.plot(x[window_size - 1:], ma, c='r',
             alpha=0.7, label="reward moving average")

    if space_visitation is not None:
        ax2 = ax.twinx()
        ax2.plot(x, space_visitation, c='black')
        ax2.set_ylabel("space visitation in %")
        ax2.set_ylim([0, 100])
    plt.tight_layout()
    return plt
   
def plot_trajectory(graph):
    graph.convert_labels('adjacency')
    adj = scipy.sparse.csr_matrix(graph.get_adjacency_matrix())
    pos = np.array([tuple(map(int, graph.index_node_labels[i].split('.')))
                    for i in range(len(graph.index_node_labels))])
    im = sknetwork.visualization.svg_graph(adj, pos, directed=True)
    return SVG(im)
def plot_intrinsic_scores_grid(data, label, num_runs, a,space_visitation=None):
    c = 2 #was 3
    r = int(round(num_runs / c + 0.49, 0))

    fig, axes = plt.subplots(r, c, figsize=(r * 6, c * 5), sharey=True)
    for n in range(num_runs):
        ax = axes[n // c][n % c]
        scores = data[(label, str(n))]
        window_size = 100
        ma = np.convolve(scores, np.ones(window_size, dtype=int), 'valid')
        ma /= window_size
        x = np.arange(len(scores))
        ax.set_xlabel("episodes", fontsize=18)
        ax.set_ylabel("rewards", fontsize=18)
        #ax.title.set_text(f'run {n}')
        ax.scatter(x, scores, marker='+', c='b', s=30, linewidth=1,
                   alpha=0.5, label="total rewards")
        ax.plot(x[window_size - 1:], ma, c='r',
                alpha=0.7, label="reward moving average")
        if space_visitation is not None:
            sv = space_visitation[(label, str(n))]
            x = np.arange(len(sv))
            ax2 = ax.twinx()
            ax2.plot(x, sv, c='black')
            ax2.set_ylabel("space visitation in %",fontsize=18)
            ax2.set_ylim([0, 100])

    for i in range(n + 1, r * c):
        ax = axes[i // c][i % c]
        ax.axis('off')

    fig.suptitle(f'Total rewards over each run for {a}', fontsize=15)
    plt.tight_layout()

    #plt.show()
    return plt