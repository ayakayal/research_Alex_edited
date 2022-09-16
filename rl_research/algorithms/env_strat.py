import numpy as np
from .neural_networks import np_nn_softmax_out
from ..utils import argmax_tiebreaker


class ES_Agent(np_nn_softmax_out):
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

    def evaluate(self, pr=False):
        state = self.env.reset()
        rewards, actions, states = [], [], []
        score = 0
        t = 0
        done = False

        while not done and t < self.env._max_episode_steps:
            action = self.get_action(state)

            states.append(state)
            actions.append(action)

            state, reward, done, _ = self.env.step(action)

            rewards.append(reward)

            score += reward

            t += 1

        self.score = score

        self.trajectory = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards)
        }

        return score

    def get_policy(self):
        self.policy = {x: self.get_action(x) for x in self.env.states}


def create_pop(env, agent, npop=2, sigma=0.1):
    new_agents = [ES_Agent(env, init_weights=agent.weights)
                  for _ in range(npop)]

    N = {}

    for key, val in agent.weights.items():
        shape = val.shape
        N[key] = np.random.randn(*shape, npop)

    for i, new_agent in enumerate(new_agents):
        for key in new_agent.weights:
            new_agent.weights[key] += sigma * N[key][:, :, i]

    return new_agents, N


def train(env, generation=300):
    inp = 2
    h1 = 64
    h2 = 64
    out = 3

    npop = 100
    alpha = 0.05
    sigma = 0.1

    agent = ES_Agent(env, inp=inp, h1=h1, h2=h2, out=out)

    print("\n" + "*" * 100)
    print("TRAINING START\n")

    for gen in range(generation):
        new_agents, N = create_pop(env, agent, npop=npop, sigma=sigma)
        R = np.zeros(npop)

        for i, n_a in enumerate(new_agents):
            R[i] = n_a.evaluate()

        print(np.unique(R, return_counts=True))

        if np.std(R) == 0:
            A = np.zeros(npop)
        else:
            A = (R - np.mean(R)) / np.std(R)

        for key in agent.weights:
            offset = alpha / (npop * sigma) * (N[key] @ A)
            agent.weights[key] += offset

        agent.evaluate()
        print("Generation:", gen, "Score:", agent.score)

    print("\n" + "*" * 100)
    print("TRAINING ENDED\n")

    return agent
