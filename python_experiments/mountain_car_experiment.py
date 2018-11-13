import gym
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
np.set_printoptions(threshold=np.nan)

env = gym.make('MountainCar-v0')

### See Gym-mountain car in action with randomly selected action. Just for visualization
for i_episode in range(50):
    observation = env.reset()
    print(observation)
    for t in range(5000):
        env.render()    
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print(observation, reward)
        if reward>0:
            print("reward",reward)
        
        if done:
            print(done, reward)
            print("Episode finished after {} timesteps".format(t+1))
            break
            
### Close the currently run Gym environment
env.close()

### Discritize the state space. Define the reward, this way of defining rewards should not be necessary, rather the reward could be taken from the samples.
def get_discretized_state(obs):
    position, velocity = 0, 0
    for index, value in enumerate(disc_position):
        if obs[0]<value:
            position = index-1
            break
    for index, value in enumerate(disc_velocity):
        if obs[1]<value:
            velocity = index-1
            break
    return (position * scale) + velocity

def get_state_position_velocity(state):
    position = state//scale
    velocity = state%scale
    return disc_position[position], disc_velocity[velocity]

def get_reward(state):
    """
    Reward is -1 until the goal position 0.5 is reached, velocity doesn't matter in computing the reward.
    
    Parameters
    -----------
    state: single value representing the discretized state
    """
    position = disc_position[int(state/scale)] # state/scale returns the discritized position
    if position >= 0.5:
        return 0
    return -1
    return position

### Implement PSRL algorithm for mountain car
def PSRL(num_states, num_actions, num_next_states, discount_factor, num_episodes, num_runs):  
    regret_psrl = np.zeros( (num_runs, num_episodes) )
    
    for m in range(num_runs):
        print("run: ", m)
        #observation = env.reset()
        # Initialize uniform Dirichlet prior
        prior = np.ones( (num_states, num_actions, num_next_states) )
        samples = np.zeros( (num_states, num_actions, num_next_states) )
        posterior = prior + samples
        # Run episodes for the PSRL
        for k in range(num_episodes):
            print("episode: ", k)
            sampled_mdp = crobust.MDP(0, discount_factor)
            
            # Compute posterior
            posterior = posterior+samples
            #print("posterior", posterior)
            
            print("build the MDP")
            for s in range(num_states):
                #print("state",s)
                
                # All the states are not reachable from current state. Rather a smaller subset of states around the current states are reachable.
                # Consider only those nearby states for possible transition to reduce the complexity and running time.
                position, velocity = get_state_position_velocity(s)
                next_states = np.unique([get_discretized_state( (p,v) ) for p in np.arange(max(position-0.3,position_lowest),\
                                        min(position+0.3,position_highest), position_step) for v in np.arange(max(velocity-0.02,velocity_lowest),\
                                                                                        min(velocity+0.02,velocity_highest),velocity_step)])
                for a in range(num_actions):
                    trp =  np.random.dirichlet(posterior[s,a], 1)[0]
                    
                    next_trp = trp[ next_states ]
                    normalizer = np.sum(next_trp)
                    
                    if s%200==0:
                        print("episode",k,"current state",s,"action",a, "sum_visits",np.sum(posterior[next_states]), "total sum", np.sum(posterior), "normalizer", normalizer)
                    
                    for s_next in next_states: #range(num_next_states):
                        sampled_mdp.add_transition(s, a, s_next, trp[s_next]/normalizer, get_reward(s_next))
            
            print("Solve the problem")
            # Compute current solution
            cur_solution = sampled_mdp.solve_mpi()
            cur_policy = cur_solution.policy
            
            #print("cur_solution", cur_solution)
            
            print("compute return and execute policy to collect samples")
            # Initial state is uniformly distributed, compute the expected value over them.
            expected_value_initial_state = 0
            for i in init_states:
                expected_value_initial_state += cur_solution[0][i]
            expected_value_initial_state /= len(init_states)
            
            regret_psrl[m,k] = expected_value_initial_state #abs(cur_solution[0][0]-true_solution[0][0])
            #print("PSRL cur_solution[0][0]",cur_solution[0][0], "Regret: ", regret_psrl[k,m])
            samples = np.zeros((num_states, num_actions, num_next_states))
            
            # Follow the policy to collect transition samples
            #done = False
            observation = env.reset()
            cur_state = get_discretized_state(observation)
            for h in range(horizon):
                action = cur_policy[cur_state]
                observation, reward, done, info = env.step(action)
                next_state = get_discretized_state(observation)
                samples[cur_state, action, next_state] += 1
                #print("cur_state", cur_state, "action", action, "next_state", next_state, "samples", samples[cur_state, action, next_state])
                cur_state = next_state
                if done:
                    print("----- destination reached in",h,"steps, done execution. -----")
                    break
    
    print("------ Interpreetd solution ------")
    for i in np.arange(position_lowest,position_highest+position_step,position_step):
        for j in np.arange(velocity_lowest, velocity_highest+velocity_step, velocity_step):
            state_index = get_discretized_state((i,j))
            #print("position", i,"velocity", j,"state index", state_index, "value function", cur_solution[0][state_index], "policy", cur_policy[state_index])
        
    #regret_psrl = np.mean(regret_psrl, axis=1)
    return np.amin(regret_psrl, axis=0), np.mean(regret_psrl, axis=0), cur_solution

### Run the learned policy and render to see it in action
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
            state =  get_discretized_state(obs)
            action = policy[state]
        obs, reward, done, info = env.step(action)
        total_reward += discount_factor ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

### Set the experiment and discritization parameters. Trigger the PSRL algorithm

position_lowest = env.observation_space.low[0]
position_highest = env.observation_space.high[0]

velocity_lowest = env.observation_space.low[1]
velocity_highest = env.observation_space.high[1]

scale = 40
disc_position = np.linspace(position_lowest, position_highest, scale)
disc_velocity = np.linspace(velocity_lowest, velocity_highest, scale)

position_step = (position_highest-position_lowest)/scale
velocity_step = (velocity_highest-velocity_lowest)/scale

# Initial state is a random position between -0.6 to -0.4 with no velocity
init_positions = np.arange(-0.6, -0.4, position_step)
init_states = [get_discretized_state((x,0)) for x in init_positions]

date_time = str(datetime.datetime.now())

num_states, num_next_states = scale*scale, scale*scale
num_actions = 3

confidence = 0.9
discount_factor = 1.0
num_episodes = 3
num_runs = 1
horizon = 1000

psrl_rets = PSRL(num_states, num_actions, num_next_states, discount_factor, num_episodes, num_runs)
policy = psrl_rets[2].policy

###
run(True, policy)

###
env.close()

### Prepare data from learned policy-valuefunction for plotting
position, velocity, policy = [], [], []

for i in np.arange(position_lowest,position_highest+position_step,position_step):
    for j in np.arange(velocity_lowest, velocity_highest+velocity_step, velocity_step):
        position.append(i)
        velocity.append(j)
        state_index = get_discretized_state((i,j))
        act = psrl_rets[2].policy[state_index]
        policy.append(act) 
        #print("position", i,"velocity", j,"state index", state_index, "value function", cur_solution[0][state_index], "policy", cur_policy[state_index])
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












