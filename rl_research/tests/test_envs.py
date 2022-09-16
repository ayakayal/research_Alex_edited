import unittest
import numpy as np
import gym
import gym_keygrid

from ..algorithms import Reinforce, ReinforceBaseline, ReinforceCountState, ReinforceCountSeq


class TestEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = [gym.make('keygrid-v0', grid_length=4),
                    gym.make('keygrid-v1', grid_length=4)]

    @classmethod
    def tearDownClass(cls):
        for env in __class__.envs:
            env.close()

    def setUp(self):
        self.envs = __class__.envs

    def test_optimalpolicy(self):
        for env in self.envs:
            done = False
            total_reward = 0
            state = env.reset()
            step = 0
            while not done and step < env._max_episode_steps:
                step += 1
                action = env.opt_policy[state]
                new_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = new_state

            self.assertTrue(done)
            self.assertEqual(total_reward, env.total_reward)


if __name__ == '__main__':
    unittest.main()
