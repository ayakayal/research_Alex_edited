import gym
from gym import error, spaces, utils
from gym.utils import seeding
import itertools
import random
import numpy as np

LEFT = -1
PICK = 0
RIGHT = 1
KEY_NOT_FOUND = 0
KEY_FOUND = 1


class KeyGridSparse(gym.Env):
    def __init__(self, grid_length=6, final_reward=50):
        self.len = grid_length

        # States definition
        position_states = [i for i in range(self.len)]
        key_states = [KEY_NOT_FOUND, KEY_FOUND]
        self.states = list(itertools.product(position_states, key_states))
        self.state_low = [min(position_states), min(key_states)]
        self.state_high = [max(position_states), max(key_states)]
        self.states_n = (len(position_states), len(key_states))
        #self.observation_space= spaces.Box(low=np.array([min(position_states), min(key_states)]), high=np.array([max(position_states), max(key_states)]))
        #self.observation_space= spaces.Tuple(spaces.Box(low=min(position_states), high=max(position_states), shape=(1,), dtype='int'),spaces.Box(low=min(key_states), high=max(key_states), shape=(1,), dtype='int'))
        #self.observation_space= spaces.Box(low=np.array([min(position_states), min(key_states)]), high=np.array([max(position_states), max(key_states)]), shape=(2,))
        self.observation_space= spaces.Box(low=min(position_states), high=max(position_states), shape=(2,))
        #self.observation_space= spaces.Dict({"position": spaces.Discrete(len(position_states),start=min(position_states)),"key": spaces.Discrete(len(key_states),start=min(key_states))})
        
        #self.observation_space= spaces.Box(low=min(position_states), high=np.array([max(position_states), max(key_states)]))
        # Actions definition
        self.actions = [LEFT, PICK, RIGHT]
        self.action_space=spaces. Discrete(3, start=-1) 
        self.actions_dic = {-1: 'LEFT', 0: 'PICK', 1: 'RIGHT'}
        self.actions_n = len(self.actions)

        self.counter = 0
        self.done = 0

        self._start_pos = min(position_states)
        self._end_pos = min(position_states)

        self._start_state = tuple([self._start_pos, KEY_NOT_FOUND])
        self._end_state = tuple([self._end_pos, KEY_FOUND])

        self._key_pos = (self.len - 1) - grid_length // 5

        self.max_episode_steps = self.len * 4

        self.reset()
        self.init_representation()

        self.intermediate_reward = 0
        self.final_reward = final_reward
        self.total_reward = self.intermediate_reward + self.final_reward
        self.init_reward()
        self.init_optimal_policy()

        print(
            f"You are using the {self.__class__.__name__} environment with length {grid_length}")

    def reset(self):
        self.state = self._start_state
        self.key_found = KEY_NOT_FOUND
        self.counter = 0
        self.done = 0
        self.reward = 0

        return self.state

    def step(self, action, render=False):
        self.counter += 1
        self.reward = 0

        if action in self.actions and not self.done:
            self.reward += self.reward_scheme[(self.state, action)]
            self.perform_action(action)

        self.done = (self.state == self._end_state)

        if render:
            self.render()
        #print (self.state,' ', self.reward,' ', self.done,' ', self.info)
        return self.state, self.reward, self.done, {}

    def perform_action(self, action):
        [pos, key] = self.state

        if action in [LEFT, RIGHT]:
            new_pos = max(min(pos+action, self.len-1), 0)
            self.state = tuple([new_pos, key])

        if action == PICK and pos == self._key_pos:
            self.state = tuple([pos, KEY_FOUND])
            self.key_found = KEY_FOUND

    def sample_action(self):
        return random.choice(self.actions)

    def get_state(self):
        return self.state

    def sample_state(self):
        return random.choice(self.states)

    def init_representation(self):
        self.representation = {i: '*' for i in range(self.len)}
        if not self.key_found:
            self.representation[self._key_pos] = 'K'

        self.representation[self.state[0]] = 'A'

    def init_reward(self):
        self.reward_scheme = {
            sa: 0 for sa in itertools.product(self.states, self.actions)}
        for a in self.actions:
            self.reward_scheme[(
                (self._end_state[0]-a, self._end_state[1]), a)] = self.final_reward

        self.reward_scheme[((self._key_pos, 0), KEY_NOT_FOUND)] = self.intermediate_reward

    def init_optimal_policy(self):
        self.opt_policy = {s: None for s in self.states}
        for j in range(self.len):
            if j < self._key_pos:
                self.opt_policy[(j, KEY_NOT_FOUND)] = RIGHT
            else:
                self.opt_policy[(j, KEY_NOT_FOUND)] = LEFT
            self.opt_policy[(j, KEY_FOUND)] = LEFT
        self.opt_policy[(self._key_pos, KEY_NOT_FOUND)] = PICK

    def render(self):
        self.init_representation()
        str = '| '
        for i in range(self.len):
            str += self.representation[i] + ' | '

        print(str)

    def close(self):
        pass
