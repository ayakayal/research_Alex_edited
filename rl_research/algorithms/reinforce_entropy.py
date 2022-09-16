import tensorflow as tf
import numpy as np
import random

import math
from .neural_networks import tf_nn_softmax_out


class Reinforce:
    def __init__(self, env, *args, **kwargs):
        self.env = env
        self.env_solved = False
        self.threshold = 10
        self.state_space = len(env.states_n)
        self.action_space = env.actions_n
        self.policy = tf_nn_softmax_out(
            *args, inp=self.state_space, out=self.action_space, **kwargs)
        self.gamma = 0.99
        self.intrinsic_reward = 1
        self.state_freq = {x: 0 for x in self.env.states}
        self.trajectories = []
        self.score = 0
        self.intrinsic_score = 0

    def sample_action(self, inp):
        scaled_inp = self.scale_state(inp)
        prob = self.policy.forward(scaled_inp)[0]
        action_idx = random.choices(range(len(prob)), weights=prob)[0]
        return action_idx, self.env.actions[action_idx]

    def get_action(self, inp):
        scaled_inp = self.scale_state(inp)
        action_idx = self.policy.predict(scaled_inp)
        return action_idx, self.env.actions[action_idx]

    def get_policy(self):
        self.pi = {x: self.get_action(x)[1] for x in self.env.states}
        return self.pi

    def evaluate_policy(self):
        pi = self.get_policy()
        render = True
        state = self.env.reset()
        score = 0
        step = 0
        done = False

        while not done and step < self.env.max_episode_steps:
            step += 1
            action = pi[state]

            state, reward, done, _ = self.env.step(action)
            score += reward

            if render:
                self.env.render()

        return reward


    def get_proba(self):
        scaled_s = self.scale_state(np.array(self.env.states))
        probas = self.policy.forward(scaled_s)
        p = {s: probas[i] for i, s in enumerate(self.env.states)}
        return p

    def get_space_visitation(self):
        space_visitation = 100 * sum([x > 1 for x in self.state_freq.values()]
                                     ) / len(self.state_freq)
        return space_visitation

    def scale_state(self, s):
        state_max = np.array(self.env.state_high)
        state_min = np.array(self.env.state_low)
        state_range = state_max - state_min
        return (np.array(s) - state_min) / state_range

    def comp_gain(self):
        for t in self.trajectories:
            if 'gains' not in t:
                r = t['rewards']
                g = np.zeros(r.shape)

                g[-1] = r[-1]
                for j in range(len(r) - 2, -1, -1):
                    g[j] = r[j] + self.gamma * g[j + 1]

                t['gains'] = g

    def add_trajectory(self, states, actions, rewards):
        trajectory = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards)
        }
        self.trajectories.append(trajectory)

    def play_ep(self, num_ep=1, render=False):

        for n in range(num_ep):
            state = self.env.reset()
            rewards, actions, states = [], [], []
            score = 0
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

                state, reward, done, _ = self.env.step(action)

            
                
                self.state_freq[state] += 1

                rewards.append(reward)
                score += reward

                if render:
                    env.render()

            self.add_trajectory(states, actions, rewards)

            self.score = score

    def update_agent(self):
        self.comp_gain()

        states = np.concatenate([t["states"] for t in self.trajectories])
        actions = np.concatenate([t["actions"] for t in self.trajectories])
        gains = np.concatenate([t["gains"] for t in self.trajectories])

        # Normalise state in 0-1 range
        states = self.scale_state(states)

        states = tf.cast(tf.convert_to_tensor(states), dtype=tf.float32)
        actions = tf.cast(tf.convert_to_tensor(actions), dtype=tf.float32)
        gains = tf.cast(tf.convert_to_tensor(gains), dtype=tf.float32)

        # Update the policy network
        self.update_policy_net(states, actions, gains)

        self.trajectories = []

    @tf.function(experimental_relax_shapes=True)
    def update_policy_net(self, states, actions, gains):
        entropy=[]
        with tf.GradientTape() as tape:
            log_prob = self.policy.distributions(states).log_prob(actions)
            p = self.get_proba()
            for s in states:
                entropy[s]=-sum(math.log(p[s])*p[s])
            tf.cast(tf.convert_to_tensor(entropy), dtype=tf.float32)
            print('log p ',log_prob)
            entropy_loss=-tf.math.reduce_mean(entropy)
            loss = -tf.math.reduce_mean(log_prob * gains)+ entropy_loss
        grads = tape.gradient(loss, self.policy.weights)
        self.policy.optimizer.apply_gradients(
            zip(grads, self.policy.model.trainable_weights))

    def train(self, num_iter=100):
        print("\n" + "*" * 100)
        print("TRAINING START\n")

        self.scores = np.zeros(num_iter)
        self.intrinsic_scores = np.zeros(num_iter)
        self.space_visitation = np.zeros(num_iter)

        for i in range(num_iter):
            print()
            print("Iteration: ", i)
            if i % 20 == 0:
                p = self.get_proba()
                for st in self.env.states:
                    print(st, "probas: ", np.round(p[st], 2))

            self.play_ep()
            print("Score:", self.score)
            space_visitation = self.get_space_visitation()
            print(f"% of visited states: {np.round(space_visitation, 1)}%")
            self.space_visitation[i] = space_visitation

            self.update_agent()
            self.scores[i] = self.score
            self.intrinsic_scores[i] = self.intrinsic_score

            if i > self.threshold and np.all(self.scores[i-self.threshold: i] == self.env.total_reward):
                self.env_solved = True
                break

        print("\n" + "*" * 100)
        print("TRAINING ENDED\n")

    def train_without_logs(self, num_iter=100):
        self.scores = np.zeros(num_iter)
        self.intrinsic_scores = np.zeros(num_iter)
        self.space_visitation = np.zeros(num_iter)
        environment_solved = False

        for i in range(num_iter):
            self.play_ep()
            self.update_agent()
            self.scores[i] = self.score
            self.intrinsic_scores[i] = self.intrinsic_score
            self.space_visitation[i] = self.get_space_visitation()

            if i > self.threshold and np.all(self.scores[i-self.threshold: i] == self.env.total_reward):
                self.env_solved = True
                break

    @staticmethod
    def reward_calc(base_reward, freq, t, alg='UCB'):
        if alg == 'UCB':
            return base_reward * np.sqrt(2 * np.log(t) / freq)
        if alg == 'MBIE-EB':
            return base_reward * np.sqrt(1 / freq)
        if alg == 'BEB':
            return base_reward / freq
        if alg == 'BEB-SQ':
            return base_reward / freq**2
