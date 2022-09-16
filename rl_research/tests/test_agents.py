import unittest
import numpy as np
import tensorflow as tf
import copy

import gym
import gym_keygrid

from ..algorithms import Reinforce, ReinforceBaseline, ReinforceCountState, ReinforceCountSeq, ReinforceSeqComp


class TestAgents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make('keygrid-v0', grid_length=4)

    @classmethod
    def tearDownClass(cls):
        __class__.env.close()

    def setUp(self):
        np.random.seed(22)

        self.env = __class__.env
        self.env.reset()
        self.agents = [Reinforce(self.env), ReinforceBaseline(
            self.env), ReinforceCountState(self.env), ReinforceCountSeq(self.env),
            ReinforceSeqComp(self.env)]

    def test_sample_actions(self):
        state = self.env.sample_state()
        for agent in self.agents:
            for _ in range(10):
                action_idx, action = agent.sample_action(state)
                self.assertIsInstance(action, int)
                self.assertIn(action, self.env.actions)

    def test_policy(self):
        state = self.env.sample_state()
        for agent in self.agents:
            for _ in range(10):
                action_idx, action = agent.get_action(state)
                self.assertIsInstance(action, int)
                self.assertIn(action, self.env.actions)

    def test_policy_updates(self):
        for agent in self.agents:
            state = self.env.sample_state()
            action_idx, action = agent.sample_action(state)
            reward = 20
            trajectory = {
                'states': np.array([state]),
                'actions': np.array([action_idx]),
                'rewards': np.array([reward])
            }
            agent.trajectories.append(trajectory)

            old_weights = copy.deepcopy(agent.policy.weights)
            agent.update_agent()
            new_weights = copy.deepcopy(agent.policy.weights)

            weights_change = True
            for ow, nw in zip(old_weights, new_weights):
                weights_change = weights_change and tf.reduce_all(
                    tf.math.equal(ow, nw))

            self.assertFalse(weights_change)

    def test_roll_out(self):
        state = self.env.sample_state()
        for agent in self.agents:
            for it in range(5):
                action_idx, action = agent.sample_action(state)
                state, _, _, _ = self.env.step(action)
                self.assertTrue(state)

    def test_easy_problem(self):
        for agent in self.agents:
            agent.train_without_logs(num_iter=150)
            score = np.mean(agent.scores)
            self.assertTrue(score > (self.env.total_reward // 5))


if __name__ == '__main__':
    unittest.main()
