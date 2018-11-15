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

date_time = str(datetime.datetime.now())

### Implement helper methods

def obs_to_index(obs):
    """ 
    Maps an observation to discrete state index for both position and velocity 
    
    Parameters
    -----------
    obs: The observation tuple
    
    Returns
    --------
    a: discritized position index, 
    b: discritized velocity index
    """
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b

def index_to_obs(a,b):
    """ 
    Maps position and velocity indices to observation 
    
    Parameters
    -----------
    a: position index
    b: velocity index
    
    Returns
    --------
    position: position value in the continuous state space
    velocity: velocity value in the continuous state space
    """
    position = grid_x[a]
    velocity = grid_y[b]
    return position, velocity
    
#print(index_to_obs(3,0))

def index_to_single_index(a,b):
    """ 
    Calculates a single unique index based on the position and velocity
    
    Parameters
    -----------
    a: position index
    b: velocity index
    
    Returns
    --------
    single unique index
    """
    return a*resolution+b

def single_index_to_index(x):
    """ 
    computes the position and velocity indices from the single unique index
    
    Parameters
    -----------
    x: unique index
    
    Returns
    --------
    position and velocity indices
    """
    return x//resolution, x%resolution

#print(index_to_single_index(2,3))
#print(single_index_to_index(23))

def set_parametric_prior(p_index, v_index, action):
    """
    Computes a parametric Dirichlet prior distribution based on current state-action values
    
    Parameters
    -----------
    p_index: position index
    v_index: velocity index
    action: action to take
    
    Returns
    --------
    prior: Dirichlet distribution for the prior
    """
    effect_left = -3 if action==0 else -2 if action==1 else -1
    effect_right = 1 if action==0 else 2 if action==1 else 3
    
    p_min = max(p_index+effect_left, 0)
    p_max = min(p_index+effect_right, resolution-1)
    
    v_min = max(v_index+effect_left, 0)
    v_max = min(v_index+effect_right, resolution-1)
    
    prior = {}
    for p in range(p_min, p_max+1):
        for v in range(v_min, v_max+1):
            normalizer = abs(p-p_index)+abs(v-v_index)+1
            prior[(p,v)] = max_prior//normalizer

    return prior

#set_parametric_prior(5, 4, 1)

def obtain_parametric_priors():
    """
    Computes a parametric Dirichlet prior distributions for all state action pairs
    
    Parameters
    -----------
    
    Returns
    --------
    priors: prior Dirichlet distribution for all state-action
    """
    priors = []
    for p in range(resolution):
        for v in range(resolution):
            for a in range(num_actions):
                priors.append(set_parametric_prior(p, v, a))
    
    priors = np.array(priors).reshape(resolution,resolution,num_actions)
    #print("priors", priors[5,5,0])
    return priors

def get_reward(state):
    """
    Reward is -1 until the goal position 0.5 is reached, velocity doesn't matter in computing the reward.
    
    Parameters
    -----------
    state: single value representing the discretized state
    
    return: reward for the corresponding state
    """
    a,b = single_index_to_index(state)
    position = index_to_obs(a, b)[0]
    if position >= 0.5:
        return 0
    return -1
#get_reward(25)
        
### Implement PSRL algorithm for mountain car
def PSRL(discount_factor, num_episodes, num_runs, horizon):
    """
    Implement posterior sampling RL algorithm for Mountain Car problem of OpenAI
    
    Parameters
    ------------
    discount_factor: discount factor for the sampled MDP
    num_episodes: number of episodes to run for the experiment
    num_runs: number of times to run the experiment
    horizon: horizon length for the execution of a policy
    
    Returns
    --------
    worst case regret, average regret and the final solution
    """
    regret_psrl = np.zeros( (num_runs, num_episodes) )
    prior = obtain_parametric_priors()
    posterior = prior
        
    for run in range(num_runs):
        print("run: ", run)
        
        # Run episodes for the PSRL
        for episode in range(num_episodes):
            print("episode: ", episode)
            sampled_mdp = crobust.MDP(0, discount_factor)
            
            print("build the MDP")
            for s in all_states:
                p,v = obs_to_index(s)
                cur_state = index_to_single_index(p,v)
                
                for action in range(num_actions):
                    samples = posterior[p,v,action]
                    
                    next_states = []
                    visit_stats = []
                
                    for key, value in samples.items():
                        next_states.append(index_to_single_index(key[0],key[1]))
                        visit_stats.append(value)
                    trp = np.random.dirichlet(visit_stats, 1)[0]
                    
                    for s_index, s_next in enumerate(next_states):
                        sampled_mdp.add_transition(cur_state, action, s_next, trp[s_index], get_reward(s_next))
            
            print("Solve the problem")
            # Compute current solution
            cur_solution = sampled_mdp.solve_mpi()
            cur_policy = cur_solution.policy
            
            print("compute return and execute policy to collect samples")
            # Initial state is uniformly distributed, compute the expected value over them.
            expected_value_initial_state = 0
            for init in init_states:
                state = index_to_single_index(init[0], init[1])
                expected_value_initial_state += cur_solution[0][state]
            expected_value_initial_state /= len(init_states)
            
            regret_psrl[run,episode] = expected_value_initial_state #abs(cur_solution[0][0]-true_solution[0][0])
            
            # Follow the policy to collect transition samples
            cur_state = obs_to_index(env.reset())
            for h in range(horizon):
                action = cur_policy[index_to_single_index(cur_state[0], cur_state[1])]
                next_state, reward, done, info = env.step(action)
                next_state = obs_to_index(next_state)
                
                # posterior[cur_position, cur_velocity, action][next_position, next_velocity] is the entry that we wanna update with the sample
                if (next_state[0],next_state[1]) not in posterior[cur_state[0],cur_state[1],action]:
                    posterior[cur_state[0],cur_state[1],action][next_state[0],next_state[1]] = 0
                posterior[cur_state[0],cur_state[1],action][next_state[0],next_state[1]] += 1
                cur_state = next_state
                
                if done:
                    print("----- destination reached in",h,"steps, done execution. -----")
                    break

    return np.amin(regret_psrl, axis=0), np.mean(regret_psrl, axis=0), cur_solution

### set experiment parameters
if __name__ == "__main__":
    resolution = 40
    max_episodes = 100#00
    discount_factor = 1.0
    eps = 0.02
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(0)
    np.random.seed(0)
    max_prior = 10
    
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / resolution
    num_actions = env.action_space.n
    
    grid_x = np.clip(np.linspace(env_low[0], env_high[0], resolution), -1.2, 0.6)
    grid_y = np.clip(np.linspace(env_low[1], env_high[1], resolution), -0.07, 0.07)
    
    all_states = np.array(list(itertools.product(grid_x, grid_y)))

    # Initial state is a random position between -0.6 to -0.4 with no velocity
    position_step = (env_high[0]-env_low[0])/resolution
    init_positions = np.arange(-0.6, -0.4, position_step)
    init_states = np.unique(np.array([obs_to_index((x,0)) for x in init_positions]), axis=0)
    
### invoke PSRL method and run experiments
if __name__ == "__main__":
    discount_factor = 1.0
    num_episodes = 50
    num_runs = 30
    horizon = 100
    
    res = PSRL(discount_factor, num_episodes, num_runs, horizon)
    print(res[2].policy)

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
position, velocity, policy = [], [], []

for i in grid_x:
    for j in grid_y:
        p,v = obs_to_index((i,j))
        position.append(i)
        velocity.append(j)
        act = res[2].policy[index_to_single_index(p,v)]
        policy.append(act) 

position, velocity, policy = np.array(position), np.array(velocity), np.array(policy)
print(len(position), len(velocity), len(policy))

### Plot the obtained policy over the discretized states
cdict = {0: 'red', 1: 'blue', 2: 'green'}
labels = ["left", "neutral", "right"]
marker = ['<','o','>']

for i in range(3):
    ix = np.where(policy == i)
    plt.scatter(position[ix], velocity[ix], c=cdict[i], label=labels[i], marker=marker[i], alpha=0.5)
plt.legend(loc='best', fancybox=True, framealpha=0.7)
plt.xlabel("position")
plt.ylabel("velocity")
#plt.colorbar()
#plt.grid()
plt.show()

### Run the obtained policy to see it in action in OpenAI

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
            state =  obs_to_index(obs)
            action = policy[index_to_single_index(state[0], state[1])]
        obs, reward, done, info = env.step(action)
        total_reward += discount_factor ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

env = wrappers.Monitor(env, "figures/mountain_car_experiments")

run(True, res[2].policy)

### Close the OpenAI experiment window
if __name__ == "__main__":
    env.close()









