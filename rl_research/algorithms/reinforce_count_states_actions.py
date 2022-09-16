import random
from collections import defaultdict

from .reinforce_baseline import ReinforceBaseline

class ReinforceCountStateAction(ReinforceBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_action_freq = defaultdict(int)
        

    def play_ep(self, num_ep=1, render=False):
        
        for n in range(num_ep):
            
          
            state = self.env.reset()
            rewards, actions, states = [], [], []
          
            score = 0
            intrinsic_score = 0
            step = 0
            done = False
            probs = self.get_proba()

            while not done and step < self.env.max_episode_steps:
                step += 1
                p = probs[state]
                action_idx = random.choices(range(len(p)), weights=p)[0]
                action = self.env.actions[action_idx]
                states.append(state)
                actions.append(action_idx)
                self.state_action_freq[tuple ([state,action])] += 1
                #print('dict ', self.state_action_freq)
                next_state, extrinsic_reward, done, _ = self.env.step(action)
                self.state_freq[next_state] += 1

                intrinsic_reward = self.reward_calc(
                    self.intrinsic_reward, self.state_action_freq[tuple ([state,action])], step, alg='MBIE-EB')

                reward = intrinsic_reward + extrinsic_reward
                rewards.append(reward)
              
                state=next_state
                score += extrinsic_reward
                intrinsic_score += intrinsic_reward

                if render:
                    self.env.render()

            self.add_trajectory(states, actions, rewards)

            self.score = score
            self.intrinsic_score = intrinsic_score


