import numpy as np
from utils import argmax_tiebreaker
from sklearn.neighbors import NearestNeighbors
from neural_networks import NNnumpy as NN


class NSES_Agent(NN):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.score = 0

    def get_action(self, state):
        if isinstance(state, tuple):
            state = np.array(state)
        if(state.shape[0] != 1):
            state = state.reshape(1, -1)

        return self.env.actions[argmax_tiebreaker(self.forward(state))]

    def evaluate(self):
        state = self.env.reset()
        score = 0
        t = 0
        done = False

        while not done and t < self.env._max_episode_steps:
            state, reward, done, _ = self.env.step(self.get_action(state))
            score += reward
            t += 1

        self.bc = (state[0] / self.env.len, state[1])
        self.score = score

        return score

    def get_policy(self):
        self.policy = {x: self.get_action(x) for x in self.env.states}


def create_pop(env, agent, npop=2, sigma=0.1):
    new_agents = [NSES_Agent(env, init_weights=agent.weights)
                  for _ in range(npop)]

    N = {}

    for key, val in agent.weights.items():
        shape = val.shape
        N[key] = np.random.randn(*shape, npop)

    for i, new_agent in enumerate(new_agents):
        for key in new_agent.weights:
            new_agent.weights[key] += sigma * N[key][:, :, i]

    return new_agents, N


def train(env):
    npop = 10
    alpha = 0.1
    sigma = 0.5
    generation = 100

    K = 2
    archive = set()
    while len(archive) != K:
        agent = NSES_Agent(env)
        agent.evaluate()
        archive.add(agent.bc)

    print("\n" + "*" * 100)
    print("TRAINING START\n")

    for gen in range(generation):
        new_agents, N = create_pop(env, agent, npop=npop, sigma=sigma)
        F = np.zeros(npop)
        R = []

        for i, n_a in enumerate(new_agents):
            F[i] = n_a.evaluate()
            R.append(n_a.bc)

        print("R: ", set(R))

        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(
            np.array(list(archive)))
        R = np.mean(nbrs.kneighbors(np.array(R))[0], axis=1)

        if np.std(R) == 0:
            A = np.zeros(npop)
        else:
            A = (R - np.mean(R)) / np.std(R)

        for key in agent.weights:
            offset = alpha / (npop * sigma) * (N[key] @ A)
            # print("offset: ", offset)
            # print("weights: ", agent.weights[key])

            agent.weights[key] += alpha / (npop * sigma) * (N[key] @ A)

        agent.evaluate()
        archive.add(agent.bc)
        print("archive: ", archive)

    print("\n" + "*" * 100)
    print("TRAINING ENDED\n")

    return agent
