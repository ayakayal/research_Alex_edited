import gym
import numpy as np

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.toy_text import discrete


LEFT = 0
RIGHT = 1


class RiverSwimEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, nS=6):
        
        # Defining the number of actions
        self.nS = nS
        self.nA = 2
        
        # Defining the reward system and dynamics of RiverSwim environment
        P, isd = self.__init_dynamics(self.nS, self.nA)
        
        super(RiverSwimEnv, self).__init__(self.nS, self.nA, P, isd)
        
        self.actions = [LEFT, RIGHT]
        self.states_n = [1]
        self.actions_n = self.nA
        self.state_high = np.array([self.nS - 1])
        self.state_low = np.array([0])
        self.states = [x for x in range(self.nS)]
        self.total_reward = 10
        
        self.render()
        
    def __init_dynamics(self, nS, nA):
        
        # P[s][a] == [(probability, nextstate, reward, done), ...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}

        # Rewarded Transitions
        P[0][LEFT] = [(1., 0, 0.005, 0)]
        P[nS-1][RIGHT] = [(0.3, nS-1, 10, 0), (0.7, nS-2, 0, 0)]

        # Left Transitions
        for s in range(1, nS):
            P[s][LEFT] = [(1., max(0, s-1), 0, 0)]

        # RIGHT Transitions
        for s in range(1, nS - 1):
            P[s][RIGHT] = [(0.3, min(nS - 1, s + 1), 0, 0), (0.6, s, 0, 0), (0.1, max(0, s-1), 0, 0)]
        
        P[0][RIGHT] = [(0.7, 0, 0, 0), (0.3, 1, 0, 0)]

        # Starting State Distribution
        isd = np.zeros(nS)
        isd[0] = 1.

        return P, isd

    def init_rep(self):
        self.representation = { i: '*' for i in range(self.nS) }
        self.representation[self.s] = 'A'

    def render(self, mode='human'):
        self.init_rep()
        str = ''
        for i in range(self.nS):
            str += '| ' + self.representation[i] + ' '
        str += '| \n'
        print(str)

    def close(self):
        pass
