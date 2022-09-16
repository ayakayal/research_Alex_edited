from collections import defaultdict
import itertools
import numpy as np
import random
from .reinforce_baseline import ReinforceBaseline


class ReinforceCountSeqStates(ReinforceBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_freq = defaultdict(int)

    def play_ep(self, num_ep=1, render=False):
        for n in range(num_ep):
            state = self.env.reset()
            rewards, actions, states = [], [], []
            score = 0
            step = 0
            done = False
            probs = self.get_proba()

            while not done and step < self.env._max_episode_steps:
                step += 1
                p = probs[state]
                action_idx = random.choices(range(len(p)), weights=p)[0]
                action = self.env.actions[action_idx]
                states.append(state)
                actions.append(action_idx)

                state, extrinsic_reward, done, _ = self.env.step(action)

                seq = tuple(itertools.chain.from_iterable(states))
                self.state_freq[state] += 1
                self.seq_freq[seq] += 1

                intrinsic_reward = self.reward_calc(self.intrinsic_reward,
                                                    self.seq_freq[seq] * self.state_freq[state], step, alg='MBIE-EB')

                reward = extrinsic_reward + intrinsic_reward

                rewards.append(reward)
                score += extrinsic_reward

                if render:
                    self.env.render()

            self.add_trajectory(states, actions, rewards)

            self.comp_gain()
            self.score = score
