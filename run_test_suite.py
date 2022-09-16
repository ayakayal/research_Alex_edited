import unittest

from rl_research.tests.test_agents import TestAgents
from rl_research.tests.test_envs import TestEnvironment
from rl_research.tests.test_dependencies import TestDependencies


def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    test_list = loader.loadTestsFromTestCase(TestAgents)
    suite.addTests(test_list)
    test_list = loader.loadTestsFromTestCase(TestEnvironment)
    suite.addTests(test_list)
    test_list = loader.loadTestsFromTestCase(TestDependencies)
    suite.addTests(test_list)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
