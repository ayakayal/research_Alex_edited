class QD_Agent(NN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inp = inp
        self.h1 = h1
        self.h2 = h2
        self.n_out = out

        self.score = 0

    def get_action(self, state):
        if isinstance(state, tuple):
            state = np.array(state)
        if(state.shape[0] != 1):
            state = state.reshape(1, -1)

        return env.actions[argmax_tiebreaker(self.forward(state))]

    def get_bc(self, env):
        state = env.reset()
        t = 0
        done = False

        while not done and t < env._max_episode_steps:
            state, reward, done, _ = env.step(self.get_action(state))
            t += 1

        bc = state
        return bc

    def evaluate(self, env):
        state = env.reset()
        score = 0
        t = 0
        done = False

        while not done and t < env._max_episode_steps:
            state, reward, done, _ = env.step(self.get_action(state))
            score += reward

            t += 1

        self.score = score

        return score


def create_pop(agent, npop=2, sigma=0.1):
    new_agents = [QD_Agent(init_weights=agent.weights) for _ in range(npop)]

    N = {}

    for key, val in agent.weights.items():
        shape = val.shape
        N[key] = np.random.randn(*shape, npop)

    for i, new_agent in enumerate(new_agents):
        for key in new_agent.weights:
            new_agent.weights[key] += sigma * N[key][:, :, i]

    return new_agents, N
