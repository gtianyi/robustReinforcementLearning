import unittest
import gym
import mountain_car_experiment_1 as mcar

### !!!! Unit tests are not implemented yet, but should have it soon

# cd /home/reazul/PhD_Research/Bayesian_Exploration/Codes/Bayes_explore/bayesian_exploration/python_experiments
class TestMountainCarMethods(unittest.TestCase):

        
    def test_obs_to_index(self):
        env_name = 'MountainCar-v0'
        env = gym.make(env_name)
        mcar.resolution = 10
        mcar.env_low = env.observation_space.low
        mcar.env_high = env.observation_space.high
        mcar.env_dx = (mcar.env_high - mcar.env_low) / mcar.resolution
        
        self.assertEqual(obs_to_index((-.5,-0.07)), (3,0))
        self.assertEqual(obs_to_index((.5,0.02)), (9,6))

suite = unittest.TestLoader().loadTestsFromTestCase(TestMountainCarMethods)
unittest.TextTestRunner(verbosity=2).run(suite)



        