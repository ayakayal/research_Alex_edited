from collections import defaultdict
import random

from .reinforce_baseline import ReinforceBaseline


class ReinforceCountSeqEdited(ReinforceBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_freq = defaultdict(int)

    def play_ep(self, num_ep=1, render=False):
        for _ in range(num_ep):
            state = self.env.reset()
            rewards, actions, states = [], [], []
         
            score = 0
            intrinsic_score = 0
            step = 0
            done = False
            seq = [] #resetting the sequence in the start of each episode
            probs = self.get_proba()

            while not done and step < self.env.max_episode_steps:
                
                step += 1
                p = probs[state]
                action_idx = random.choices(range(len(p)), weights=p)[0]
                action = self.env.actions[action_idx]
                states.append(state)
                actions.append(action_idx)
               
                
                if tuple([state,action]) not in seq:
                    seq.append(tuple ([state,action]))
                    #print('the sequence is ',seq)
                    # for s in seq:
                    #     print('elt ',s,' ')
                    self.seq_freq[tuple(seq)] += 1
                    #print('freq ',self.seq_freq)
                    intrinsic_reward = self.reward_calc(
                    self.intrinsic_reward, self.seq_freq[tuple(seq)],
                    step, alg='MBIE-EB')
                else:
                    intrinsic_reward = 0

                state, extrinsic_reward, done, _ = self.env.step(action)

                self.state_freq[state] += 1

                #if tuple([state,action]) not in seq:
                #seq+=tuple ([state,action])
                #print('the sequence is ',seq)
                #self.seq_freq[tuple(seq)] += 1
                # intrinsic_reward = self.reward_calc(
                #     self.intrinsic_reward, self.seq_freq[tuple(seq)],
                #     step, alg='MBIE-EB')
                # else:
                #     intrinsic_reward = 0

                reward = extrinsic_reward + intrinsic_reward
                rewards.append(reward)
              

                score += extrinsic_reward
                intrinsic_score += intrinsic_reward

                if render:
                    self.env.render()

            self.add_trajectory(states, actions, rewards)

            self.score = score
            self.intrinsic_score = intrinsic_score
