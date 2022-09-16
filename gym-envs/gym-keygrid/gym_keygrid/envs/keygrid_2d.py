import gym
from gym import error, spaces, utils
from gym.utils import seeding
import itertools
from itertools import product
import random

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3
PICK = 4
KEY_NOT_FOUND = 0
KEY_FOUND = 1


class KeyGrid2d(gym.Env):
    def __init__(self, grid_length=6):
        self.len = grid_length
        self.grid = list(product(range(self.len), repeat=2))
        self.states = [(*x, k)
                       for x in self.grid for k in [KEY_NOT_FOUND, KEY_FOUND]]
        self.states_n = (self.len, self.len, 2)
        self.state_low = [0, 0, 0]
        self.state_high = [self.len-1, self.len-1, KEY_FOUND]

        self.actions = [LEFT, UP, RIGHT, DOWN, PICK]
    
        self.action_space=spaces.Box(low=-1, high=1, shape=(2,))
        self.actions_dic = {LEFT: 'LEFT', UP: 'UP',
                            RIGHT: 'RIGHT', DOWN: 'DOWN', PICK: 'PICK'}
        
        self.actions_offset = {
            LEFT: (-1, 0),
            UP: (0, -1), 
            RIGHT: (1, 0),
            DOWN: (0, 1),
            PICK: (0, 0)
        }
        self.actions_n = len(self.actions)

        self.counter = 0
        self.done = 0

        self.start_pos = (0, 0)
        self.end_pos = (0, 0)

        self.initial_state = (*self.start_pos, KEY_NOT_FOUND)
        self.end_state = (*self.end_pos, KEY_FOUND)
        self.observation_space= spaces.Box(low=0, high=self.len-1, shape=(3,))
        self.key_pos = tuple([(grid_length-1) - grid_length//5]*2)

        self.max_episode_steps = self.len*10

        self.reset()
        self.init_rep()

        self.intermediate_reward = 0
        self.final_reward = 50
        self.total_reward = self.intermediate_reward + self.final_reward
        self.init_reward()
        self.comp_optimal_policy()

        print(
            f"You are using the {self.__class__.__name__} environment with length {grid_length}")


    def reset(self):
        self.state = self.initial_state
        self.key_found = KEY_NOT_FOUND
        self.counter = 0
        self.done = 0
        self.reward = 0

        return self.state

    def step(self, action, render=False):
        if action in self.actions and not self.done:
            self.reward = self.reward_scheme[(*self.state, action)]
            self.perform_action(action)
        else:
            self.reward = 0

        self.counter += 1

        if render:
            self.render()

        if self.state == self.end_state:
            self.done = 1

        return self.state, self.reward, self.done, {}

    def perform_action(self, action):
        pos = self.state[:-1]
        key = self.state[-1]

        if action in self.actions[:-1]:
            new_pos = self.get_newpos(self.state, action)
            self.state = (*new_pos, key)

        if action == PICK and pos == self.key_pos:
            self.state = (*pos, KEY_FOUND)
            self.key_found = KEY_FOUND

    def init_rep(self):
        self.representation = {
            i: {j: '*' for j in range(self.len)} for i in range(self.len)}
        if not self.key_found:
            self.representation[self.key_pos[0]][self.key_pos[1]] = 'K'

        self.representation[self.state[0]][self.state[1]] = 'A'

    def init_reward(self):
        self.reward_scheme = {
            (*s, a): 0 for s in self.states for a in self.actions}
        for a in self.actions:
            x_offset, y_offset = self.actions_offset[a]
            reward_state = (
                self.end_state[0]-x_offset, self.end_state[1]-y_offset, self.end_state[2])
            if reward_state in self.states:
                self.reward_scheme[(*reward_state, a)] = self.final_reward

        self.reward_scheme[(*self.key_pos, KEY_NOT_FOUND, PICK)] = self.intermediate_reward

    def render(self):
        self.init_rep()
        str = ''
        for i in range(self.len):
            str += '| '
            for j in range(self.len):
                str += self.representation[j][i] + ' | '
            str += '\n'

        print(str)

    def get_newpos(self, pos, action):
        x_offset, y_offset = self.actions_offset[action]
        new_pos = (
            max(min(pos[0]+x_offset, self.len-1), 0),
            max(min(pos[1]+y_offset, self.len-1), 0),
        )
        return new_pos

    def close(self):
        pass

    def get_state(self):
        return self.state

    def sample_state(self):
        return random.choice(self.states)

    def sample_action(self):
        return random.choice(self.actions)

    def comp_optimal_policy(self):
        self.opt_policy = {s: 4 for s in self.states}
        for i in range(self.len):
            for j in range(self.len):
                if i < self.key_pos[0]:
                    self.opt_policy[(i, j, KEY_NOT_FOUND)] = RIGHT
                    self.opt_policy[(i, j, KEY_FOUND)] = LEFT

                elif j <= self.key_pos[1]:
                    self.opt_policy[(i, j, KEY_NOT_FOUND)] = DOWN
                    self.opt_policy[(i, j, KEY_FOUND)] = UP

                if j == 0:
                    self.opt_policy[(i, j, KEY_FOUND)] = LEFT

        self.opt_policy[(*self.key_pos, KEY_NOT_FOUND)] = PICK
