import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt
from utils import *
np.set_printoptions(threshold=np.nan)

class QLearning:
    def __init__(self, env, resolution, max_episodes=10000, max_iterations=10000, discount_factor=1.0):
        self.env = env
        self.resolution = resolution
        self.n_states = resolution
        self.max_episodes = max_episodes
        self.max_iterations = max_iterations
        self.initial_lr = 1.0 #Initial Learning rate
        self.min_lr = 0.003
        self.discount_factor = discount_factor
        self.eps = 0.02
        
        self.env_low = self.env.observation_space.low
        self.env_high = self.env.observation_space.high
        self.env_dx = (self.env_high - self.env_low) / self.resolution
        self.num_actions = self.env.action_space.n
        
        # Initial state is a random position between -0.6 to -0.4 with no velocity, construct the corresponding discritized initial states.
        self.position_step = (self.env_high[0]-self.env_low[0])/self.resolution
        self.init_positions = np.arange(-0.6, -0.4, self.position_step)
        self.init_states = np.unique(np.array([obs_to_index((x,0), self.env_low, self.env_dx) for x in self.init_positions]), axis=0)
        
        print("init_states", self.init_states)
        
        self.env.seed(0)
        np.random.seed(0)
        
    def train(self, render=False):
        q_table = np.zeros((self.n_states, self.n_states, self.env.action_space.n))
        for ep in range(self.max_episodes):
            obs = self.env.reset()
            total_reward = 0
            # eta: learning rate is decreased at each step
            eta = max(self.min_lr, self.initial_lr * (0.85 ** (ep//100)))
            for iter in range(self.max_iterations):
                if render:
                    env.render()
                state = self.obs_to_state(obs)
                if np.random.uniform(0, 1) < self.eps:
                    action = np.random.choice(self.env.action_space.n)
                else:
                    logits = q_table[state]
                    logits_exp = np.exp(logits)
                    probs = logits_exp / np.sum(logits_exp)
                    action = np.random.choice(self.env.action_space.n, p=probs)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                # update q table
                new_state = self.obs_to_state(obs)
                q_table[state + (action,)] = q_table[state + (action,)] + eta * (reward + self.discount_factor *  np.max(q_table[new_state]) - q_table[state + (action, )])
                if done:
                    break
            if ep % 100 == 0:
                print('Iteration #{} -- Total reward = {}.'.format(ep+1, total_reward))
        return q_table

    def run(self, render=True, policy=None):
        obs = self.env.reset()
        total_reward = 0
        step_idx = 0
        for iter in range(self.max_iterations):
            if render:
                env.render()
            if policy is None:
                action = self.env.action_space.sample()
            else:
                state =  self.obs_to_state(obs)
                action = policy[state]
            obs, reward, done, info = self.env.step(action)
            total_reward += self.discount_factor ** step_idx * reward
            step_idx += 1
            if done:
                break
        return total_reward
    
    def obs_to_state(self, obs):
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_dx = (env_high - env_low) / self.n_states
        a = int((obs[0] - env_low[0])/env_dx[0])
        b = int((obs[1] - env_low[1])/env_dx[1])
        return a, b

    def train_qlearning(self):
        q_table = self.train(render=False)
        print("q_table.shape", q_table.shape)
        solution_policy = np.argmax(q_table, axis=2)
        
        init_ret = 0.0
        for state in self.init_states:
            init_ret += np.mean(q_table[state[0],state[1]])
            #np.mean(q_table, axis=2)
            print("action values", q_table[state[0],state[1]], "mean", np.mean(q_table[state[0],state[1]]))
        init_ret /= len(self.init_states)
        
        print("init_ret", init_ret)
        
        #print("Q-table", q_table)
        return init_ret, q_table, solution_policy

### Train Q-learning
if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    resolution = 30
    
    q_learn = QLearning(env, resolution, 1000, 100)
    init_ret, q_table, solution_policy = q_learn.train_qlearning()

### Run & animate the computed policy in OpenAI environment
if __name__ == '__main__':
    # Animate it
    solution_policy_scores = [q_learn.run(render=False, policy=solution_policy) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    q_learn.run(render=True, policy=solution_policy)

### Close the OpenAI animation window
if __name__ == '__main__':
    env.close()

### Prepare data from learned policy-valuefunction for plotting
if __name__ == '__main__':
    position_lowest = env.observation_space.low[0]
    position_highest = env.observation_space.high[0]
    
    velocity_lowest = env.observation_space.low[1]
    velocity_highest = env.observation_space.high[1]
    
    position_step = (position_highest-position_lowest)/q_learn.n_states
    velocity_step = (velocity_highest-velocity_lowest)/q_learn.n_states
    
    position, velocity, policy = [], [], []
    
    for i in np.arange(position_lowest,position_highest, position_step):
        for j in np.arange(velocity_lowest, velocity_highest, velocity_step):
            position.append(i)
            velocity.append(j)
            state = q_learn.obs_to_state((i,j))
            act = solution_policy[state]
            policy.append(act) 
            #print("position", i,"velocity", j,"state index", state_index, "value function", cur_solution[0][state_index], "policy", cur_policy[state_index])
    position, velocity, policy = np.array(position), np.array(velocity), np.array(policy)
    print(len(position), len(velocity), len(policy))

### Plot the obtained policy over the discretized states
if __name__ == '__main__':
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

