
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt

###
resolution = 10
max_episodes = 100#00
initial_lr = 1.0 #Initial Learning rate
min_lr = 0.003
discount_factor = 1.0
max_iterations = 10000
eps = 0.02
env_name = 'MountainCar-v0'
env = gym.make(env_name)
env.seed(0)
np.random.seed(0)

env_low = env.observation_space.low
env_high = env.observation_space.high
env_dx = (env_high - env_low) / resolution

###
grid_x = np.linspace(env_low[0]-0.1, env_high[0]+0.1, resolution)
grid_y = np.linspace(env_low[1]-0.01, env_high[1]+0.01, resolution)
max_prior = 20

def obs_to_index(obs):
    """ Maps an observation to state index """
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b
    
def index_to_obs(a,b):
    """ Maps an index to observation """
    position = grid_x[a]
    velocity = grid_y[b]
    return position, velocity
    
print(obs_to_index((-.5,-0.07)))
print(index_to_obs(3,0))

def set_parametric_prior(p_index, v_index, action):
    effect_left = -3 if action==0 else -2 if action==1 else -1
    effect_right = 1 if action==0 else 2 if action==1 else 3
    
    p_min = max(p_index+effect_left, 0)
    p_max = min(p_index+effect_right, resolution-1)
    
    v_min = max(v_index+effect_left, 0)
    v_max = min(v_index+effect_right, resolution-1)
    
    print(p_min, p_max, v_min, v_max)
    
    prior = {}
    for p in range(p_min, p_max+1):
        for v in range(v_min, v_max+1):
            normalizer = abs(p-p_index)+abs(v-v_index)+1
            prior[(p,v)] = max_prior//normalizer
            print(p,v,prior[(p,v)])
    #print(prior)
    return prior

set_parametric_prior(5, 4, 1)

###
resolution = 10
#grid_x, grid_y = np.mgrid[ env_low[0]-0.1 : env_high[0]+0.1 : resolution, env_low[1]-0.01 : env_high[1]+0.01 : resolution ]

print(grid_x, grid_y)

###
for i_episode in range(1):
    cur_state = env.reset()

    for t in range(5000):
        env.render()    
        action = 2#env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(cur_state, action, next_state)
        cur_state = next_state        
        if done:
            print(done, reward)
            print("Episode finished after {} timesteps".format(t+1))
            break


###
env.close()

