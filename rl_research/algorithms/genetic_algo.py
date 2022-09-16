import numpy as np
from .neural_networks import np_nn_softmax_out
from ..utils import argmax_tiebreaker

class GA_Agent:
    def __init__(self, env, inp=2, h1=256, h2=256, out=3, mu_prob=0.3):
        self.env = env

        self.inp = inp
        self.h1 = h1
        self.h2 = h2
        self.n_out = out
        self.mu_prob = mu_prob

        self.net = np_nn_softmax_out(inp, h1, h2, out)
        self.score = 0

    def reborn(self, parents):
        parent1, parent2 = parents
        for key in self.net.weights:
            mask = np.random.choice(
                [0, 1], size=self.net.weights[key].shape, p=[.5, .5])
            self.net.weights[key] = np.where(
                mask == 1, parent1.net.weights[key], parent2.net.weights[key])

    def mutate(self):
        for key in self.net.weights:
            mask = np.random.choice([0, 1], size=self.net.weights[key].shape, p=[
                                    1 - self.mu_prob, self.mu_prob])

            random = np_nn_softmax_out.xavier_init(
                mask.shape[0], mask.shape[1])
            self.net.weights[key] = np.where(
                mask == 1, self.net.weights[key] + random, self.net.weights[key])

    def get_action(self, state):
        if isinstance(state, tuple):
            state = np.array(state)
        if state.shape[0] != 1:
            state = state.reshape(1, -1)

        return self.env.actions[argmax_tiebreaker(self.net.forward(state))]

    def evaluate(self, env, pr=False):
        state = env.reset()
        score = 0
        t = 0
        done = False

        while not done and t < env._max_episode_steps:
            state, reward, done, _ = env.step(self.get_action(state))
            if pr:
                print(state, self.get_action(state), reward)
            score += reward

            t += 1

        self.score = score

        return score

    def get_policy(self):
        self.policy = {x: self.get_action(x) for x in self.env.states}


def select_next_gen(agent_set, n_selected):
    n_best = int(n_selected * 0.8)
    n_random = n_selected - n_best

    sorted_agents = sorted(
        agent_set, key=lambda agent: agent.score, reverse=True)

    next_gen = sorted_agents[:n_best]
    next_random = np.random.choice(sorted_agents, size=n_random, replace=False)
    for rand in next_random:
        next_gen.append(rand)
    not_selected = []
    for agent in agent_set:
        if agent not in next_gen:
            not_selected.append(agent)
    return sorted_agents[0], next_gen, not_selected


def select_parents(agent_set):
    return np.random.choice(agent_set, size=2, p=np_nn_softmax_out.softmax(
        [agent.score for agent in agent_set]))


def get_best_agent(best_agents):
    return max(best_agents, key=lambda x: x.score)


def train(env, generation=100):
    inp = 2
    h1 = 64
    h2 = 64
    out = 3
    mu_prob = 0.2
    population = 30

    prop_selected = 0.3

    n_selected = int(prop_selected * population)
    best_agents = []

    agent_set = [GA_Agent(env, inp, h1, h2, out, mu_prob)
                 for i in range(population)]

    best, selected, next_children = select_next_gen(agent_set, n_selected)

    best_agents.append(best)
    score_evolution = []
    time_evolution = []

    print("\n" + "*" * 100)
    print("TRAINING START\n")

    for gen in range(1, generation + 1):
        for new_child in next_children:
            parents = select_parents(selected)
            new_child.reborn(parents)
            new_child.mutate()

        for agent in agent_set:
            agent.evaluate(env)

        best, selected, next_children = select_next_gen(agent_set, n_selected)
        best_agents.append(best)

        print("Generation:", gen, "Score:", best.score)
        score_evolution.append(best.score)

    print("\n" + "*" * 100)
    print("TRAINING ENDED\n")

    return best_agents
