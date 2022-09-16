import unittest


class TestDependencies(unittest.TestCase):
    def test__gym(self):
        import gym
        return self.assertIsNotNone(gym)

    def test__numpy(self):
        import numpy as np
        return self.assertIsNotNone(np)

    def test__scipy(self):
        import scipy
        return self.assertIsNotNone(scipy)

    def test__pandas(self):
        import pandas as pd
        return self.assertIsNotNone(pd)

    def test__mpl(self):
        import matplotlib as mpl
        mpl.use('Agg')
        return self.assertIsNotNone(mpl)

    def test__tensorflow(self):
        import tensorflow
        return self.assertIsNotNone(tensorflow)


if __name__ == '__main__':
    unittest.main()
