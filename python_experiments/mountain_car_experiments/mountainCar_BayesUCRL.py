import gym
from gym import wrappers
import random
import math
import numpy as np
from scipy.stats import norm
from craam import crobust
from scipy import stats
import matplotlib.pyplot as plt
#import tqdm
import pickle
import datetime
import itertools
import datetime
from scipy.interpolate import griddata
from utils import *

date_time = str(datetime.datetime.now())

# cd /home/reazul/PhD_Research/Bayesian_Exploration/Codes/Bayes_explore/bayesian_exploration/python_experiments/mountain_car_experiments
class BayesUCRL:
    def __init__(self, env, resolution, num_runs = 10, num_episodes = 10, horizon = 10, discount_factor = 1.0):
        self.env = env
        self.resolution = resolution
        
        self.discount_factor = discount_factor
        self.num_runs = num_runs
        self.num_episodes = num_episodes
        
        # horizon for policy execution to collect samples for the next episode
        self.horizon = horizon
        
        self.env_low = self.env.observation_space.low
        self.env_high = self.env.observation_space.high
        self.env_dx = (self.env_high - self.env_low) / self.resolution
        self.num_actions = self.env.action_space.n
        
        # discritize the whole state space, both for position and velocity
        self.grid_x = np.clip(np.linspace(self.env_low[0], self.env_high[0], self.resolution), -1.2, 0.6)
        self.grid_y = np.clip(np.linspace(self.env_low[1], self.env_high[1], self.resolution), -0.07, 0.07)
        
        # enumerate all possible discritized states
        self.all_states = np.array(list(itertools.product(self.grid_x, self.grid_y)))
        
        # Initial state is a random position between -0.6 to -0.4 with no velocity, construct the corresponding discritized initial states.
        self.position_step = (self.env_high[0]-self.env_low[0])/self.resolution
        self.init_positions = np.arange(-0.6, -0.4, self.position_step)
        self.init_states = np.unique(np.array([obs_to_index((x,0), self.env_low, self.env_dx) for x in self.init_positions]), axis=0)
        
        self.env.seed(0)
        np.random.seed(0)
        
    def train(self, num_bayes_samples, q_init_ret):
        """
        Implements the Bayes UCRL idea. Computes ambiguity set from posterior samples for required confidence levels.
            
        Returns
        --------
        numpy array
            Computed regret
        """  
        #num_bayes_samples = 20
        
        regret_bayes_ucrl = np.zeros( (self.num_runs, self.num_episodes) )
        prior = obtain_parametric_priors(self.resolution, self.num_actions)
    
        # initially the posterior is the same as prior
        posterior = prior
        
        for run in range(self.num_runs):
            for episode in range(self.num_episodes):
                sampled_mdp = crobust.MDP(0, self.discount_factor)
                
                # Compute posterior
                #posterior = posterior+samples
                thresholds = [[] for _ in range(3)]
                
                num_states = self.resolution*self.resolution
                confidence = 1-1/(episode+1)
                sa_confidence = 1-(1-confidence)/(num_states*self.num_actions) # !!! Apply union bound to compute confidence for each state-action
                
                # iterate over all state-actions, sample from the posterior distribution and construct the sampled MDP
                for s in self.all_states:
                    p,v = obs_to_index(s, self.env_low, self.env_dx)
                    cur_state = index_to_single_index(p,v, self.resolution)
                    
                    for action in range(self.num_actions):
                        samples = posterior[p,v,action]
                        
                        next_states = []
                        visit_stats = []
                        
                        # unbox key(next states) and values(Dirichlet prior parameters) from the samples dictionary.
                        for key, value in samples.items():
                            next_states.append(index_to_single_index(key[0],key[1], self.resolution))
                            visit_stats.append(value)
                        
                        # sample from the drichlet distribution stated with the prior parameters
                        # trp = np.random.dirichlet(visit_stats, 1)
                        
                        bayes_samples =  np.random.dirichlet(visit_stats, num_bayes_samples)
                        nominal_point_bayes = np.mean(bayes_samples, axis=0)
                        nominal_point_bayes /= np.sum(nominal_point_bayes)
                        
                        bayes_threshold = compute_bayesian_threshold(bayes_samples, nominal_point_bayes, sa_confidence)
                        
                        for s_index, s_next in enumerate(next_states):
                            sampled_mdp.add_transition(cur_state, action, s_next, nominal_point_bayes[s_index], get_reward(s_next, self.resolution, self.grid_x, self.grid_y))
                        
                        # construct the threshold for each state-action
                        thresholds[0].append(cur_state) # from state
                        thresholds[1].append(action) # action
                        thresholds[2].append(bayes_threshold) # allowed deviation
                
                # Compute current solution
                cur_solution = sampled_mdp.rsolve_mpi(b"optimistic_l1",np.array(thresholds))
                cur_policy = cur_solution.policy
                
                # Initial state is uniformly distributed, compute the expected value over them.
                expected_value_initial_state = 0
                for init in self.init_states:
                    state = index_to_single_index(init[0], init[1], self.resolution)
                    expected_value_initial_state += cur_solution[0][state]
                expected_value_initial_state /= len(self.init_states)
                
                # regret computation needs to be implemented. The Q-learning solution can be considered as the true soultion.
                # the solution produced here by PSRL is the approximate solution and the difference between them is the regret
                regret_bayes_ucrl[run,episode] = abs(q_init_ret-expected_value_initial_state) #abs(cur_solution[0][0]-true_solution[0][0])
                
                # Follow the policy to collect transition samples
                cur_state = obs_to_index(self.env.reset(), self.env_low, self.env_dx)
                for h in range(self.horizon):
                    action = cur_policy[index_to_single_index(cur_state[0], cur_state[1], self.resolution)]
                    next_state, reward, done, info = self.env.step(action)
                    next_state = obs_to_index(next_state, self.env_low, self.env_dx)
                    
                    # posterior[cur_position, cur_velocity, action][next_position, next_velocity] is the entry that we wanna update 
                    # with the sample. This is really combining the current sample with the prior, which constitutes the posterior.
                    if (next_state[0],next_state[1]) not in posterior[cur_state[0],cur_state[1],action]:
                        posterior[cur_state[0],cur_state[1],action][next_state[0],next_state[1]] = 0
                    posterior[cur_state[0],cur_state[1],action][next_state[0],next_state[1]] += 1
                    cur_state = next_state
                    
                    if done:
                        print("----- destination reached in",h,"steps, done execution. -----")
                        break
        
        return np.amin(regret_bayes_ucrl, axis=0), np.mean(regret_bayes_ucrl, axis=0), cur_solution
    
###
### set experiment parameters
if __name__ == "__main__": 
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    discount_factor = 1.0
    
    resolution = 20
    # number of times to run the experiment
    num_runs = 3
    
    # number of episodes for each run
    num_episodes = 3
    
    # horizon for policy execution to collect samples for the next episode
    horizon = 10
    num_bayes_samples = 20
    
    bucrl_learn = BayesUCRL(env, resolution, num_runs, num_episodes, horizon)
    
    res = bucrl_learn.train(num_bayes_samples, 0.0)
    print(res[2])

### save the result
if __name__ == "__main__":
    with open('dumped_results/mountain_car_PSRL_Policy-'+date_time,'wb') as fp:
        pickle.dump(res[2].policy, fp)
        
    with open('dumped_results/mountain_car_PSRL_ValueFunction-'+date_time,'wb') as fp:
        pickle.dump(res[2].policy, fp)
        
### load a specific result file
if __name__ == "__main__":
    f = open('dumped_results/mountain_car_policy_PSRL_parametric_prior-2018-11-15 10:01:12.820398', 'rb')
    res_check = pickle.load(f)
    print(res_check)
    
### Prepare data from learned policy-valuefunction for plotting
if __name__ == "__main__":
    position, velocity, policy = [], [], []
    
    for i in bucrl_learn.grid_x:
        for j in bucrl_learn.grid_y:
            p,v = obs_to_index((i,j), bucrl_learn.env_low, bucrl_learn.env_dx)
            position.append(i)
            velocity.append(j)
            act = res[2].policy[index_to_single_index(p,v, resolution)]
            policy.append(act) 
    
    position, velocity, policy = np.array(position), np.array(velocity), np.array(policy)
    print(len(position), len(velocity), len(policy))

### Plot the obtained policy over the discretized states
if __name__ == "__main__":
    cdict = {0: 'red', 1: 'blue', 2: 'green'}
    labels = ["left", "neutral", "right"]
    marker = ['<','o','>']
    
    for i in range(3):
        ix = np.where(policy == i)
        plt.scatter(position[ix], velocity[ix], c=cdict[i], label=labels[i], marker=marker[i], alpha=0.5)
    plt.legend(loc='best', fancybox=True, framealpha=0.7)
    plt.xlabel("position")
    plt.ylabel("velocity")
    plt.show()

### Run the obtained policy to see it in action in OpenAI
if __name__ == "__main__":
    max_iterations = 10000
    def run(render=True, policy=None):
        
        obs = env.reset()
        total_reward = 0
        step_idx = 0
        for iter in range(max_iterations):
            if render:
                env.render()
            if policy is None:
                action = env.action_space.sample()
            else:
                state =  obs_to_index(obs, bucrl_learn.env_low, bucrl_learn.env_dx)
                action = policy[index_to_single_index(state[0], state[1], resolution)]
            obs, reward, done, info = env.step(action)
            total_reward += discount_factor ** step_idx * reward
            step_idx += 1
            if done:
                break
        return total_reward

    # Save the animation
    #env = wrappers.Monitor(env, "figures/mountain_car_experiments")

    run(True, res[2].policy)

### Close the OpenAI experiment window
if __name__ == "__main__":
    env.close()
    






