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
from craam import crobust
from utils import *

date_time = str(datetime.datetime.now())

# cd /home/reazul/PhD_Research/Bayesian_Exploration/Codes/Bayes_explore/bayesian_exploration/python_experiments/mountain_car_experiments
class OFVF:
    def __init__(self, env, resolution, num_runs = 10, num_episodes = 10, horizon = 10, discount_factor = 1.0):
        self.env = env
        self.resolution = resolution
        self.num_states = self.num_next_states = self.resolution*self.resolution
        
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
        #num_bayes_samples = 20
        num_update = 10
        
        confidences = []
        regret_OFVF = np.zeros( (self.num_runs, self.num_episodes) )
        violations = np.zeros( (self.num_runs, self.num_episodes) )
        
        for m in range(self.num_runs):
            prior = obtain_parametric_priors(self.resolution, self.num_actions)
        
            # initially the posterior is the same as prior
            posterior = prior
            
            # Run episodes for the PSRL
            for k in range(self.num_episodes):
                sampled_mdp = crobust.MDP(0, self.discount_factor)
                
                confidence = 1-1/(k+1)
                sa_confidence = 1-(1-confidence)/(self.num_states*self.num_actions) # !!! Apply union bound to compute confidence for each state-action
                if m==0:
                    confidences.append(confidence)
                
                # Compute posterior
                #posterior = posterior+samples
                thresholds = [[] for _ in range(3)]
                
                posterior_transition_points = {}
                for s in self.all_states:
                    #print("--- state ---", s)
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
                                            
                        bayes_samples = np.random.dirichlet(visit_stats, num_bayes_samples)
                        posterior_transition_points[(cur_state,action)] = (bayes_samples, next_states)
                        
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
                cur_solution = sampled_mdp.rsolve_mpi(b"optimistic_l1",np.array(thresholds)) # solve_mpi()
                cur_policy = cur_solution.policy
                
                rsol = self.Optimism_VF(cur_solution[0], posterior_transition_points, num_update, sa_confidence)
                
                # Initial state is uniformly distributed, compute the expected value over them.
                expected_value_initial_state = 0
                for init in self.init_states:
                    state = index_to_single_index(init[0], init[1], self.resolution)
                    expected_value_initial_state += rsol[0][state]
                expected_value_initial_state /= len(self.init_states)
                
                regret_OFVF[m,k] = abs(q_init_ret-expected_value_initial_state)
                
                violations[m,k] = q_init_ret - expected_value_initial_state
                
                rpolicy = rsol.policy
    
                #samples = np.zeros((num_states, num_actions, num_next_states))
    
                # Follow the policy to collect transition samples
                cur_state = obs_to_index(self.env.reset(), self.env_low, self.env_dx)
                for h in range(self.horizon):
                    action = rpolicy[index_to_single_index(cur_state[0], cur_state[1], self.resolution)]
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
    
        #regret_OFVF = np.mean(regret_OFVF, axis=1)
        #plt.plot(np.cumsum(regret_OFVF))
        #plt.show()
        
        violations = np.mean(violations<0, axis=0)
        
        return np.amin(regret_OFVF, axis=0), np.mean(regret_OFVF, axis=0), violations, confidences, cur_solution
    
    def Optimism_VF(self, valuefunctions, posterior_transition_points, num_update, sa_confidence):
        """
        Method to incrementally improve value function by adding the new value function with 
        previous valuefunctions, finding the nominal point & threshold for this cluster of value functions
        with the required sa-confidence.
        
        @value_function The initially known value function computed from the true MDP
        @posterior_transition_points The posterior transition points obtained from the Bayesian sampling, 
                                        nominal point & threshold to be computed
        @num_update Number of updates over the value functions
        @sa_confidence Required confidence for each state-action computed from the Union Bound
        @orig_sol The solution to the estimated true MDP
        
        @return valuefunction The updated final value function
        """
        horizon = 1
        #s = 0
        valuefunctions = [valuefunctions]
    
        #Store the nominal points for each state-action pairs
        nomianl_points = {}
        
        #Store the latest nominal of nominal point & threshold
        nominal_threshold = {}
        under_estimate, real_regret = 0.0, 0.0
        i=0
        while i <= num_update:
            #try:
            #keep track whether the current iteration keeps the mdp unchanged
            is_mdp_unchanged = True
            threshold = [[] for _ in range(3)]
            rmdp = crobust.MDP(0, self.discount_factor)
            
            for s in self.all_states:
                #hashable_state_index = totuple(s)
                p,v = obs_to_index(s, self.env_low, self.env_dx)
                state_index = index_to_single_index(p, v, self.resolution)
                for a in range(self.num_actions):
                    
                    bayes_points = np.asarray(posterior_transition_points[state_index,a][0])
                    next_states = np.asarray(posterior_transition_points[state_index,a][1])
                        
                    RSVF_nomianlPoints = []
                    
                    #for bayes_points in trans:
                    #print("bayes_points", bayes_points, "next_states", next_states)
                    ivf = construct_uset_known_value_function(bayes_points, valuefunctions[-1], sa_confidence, next_states)
                    RSVF_nomianlPoints.append(ivf[2])
                    new_trp = np.mean(RSVF_nomianlPoints, axis=0)
                    
                    if (state_index,a) not in nomianl_points:
                        nomianl_points[(state_index,a)] = []
                    
                    trp, th = None, 0
                    #If there's a previously constructed L1 ball. Check whether the new nominal point
                    #resides outside of the current L1 ball & needs to be considered.
                    if (state_index,a) in nominal_threshold:
                        old_trp, old_th = nominal_threshold[(state_index,a)][0], nominal_threshold[(state_index,a)][1]
                        
                        #Compute the L1 distance between the newly computed nominal point & the previous 
                        #nominal of nominal points
                        new_th = np.linalg.norm(new_trp - old_trp, ord = 1)
                        
                        #If the new point is inside the previous L1 ball, don't consider it & proceed with
                        #the previous trp & threshold
                        if  (new_th - old_th) < 0.0001:
                            trp, th = old_trp, old_th
                    
                    #Consider the new nominal point to construct a new uncertainty set. This block will
                    #execute if there's no previous nominal_threshold entry or the new nominal point
                    #resides outside of the existing L1 ball
                    if trp is None:
                        is_mdp_unchanged = False
                        nomianl_points[(state_index,a)].append(new_trp)
                        
                        #Find the center of the L1 ball for the nominal points with different 
                        #value functions
                        trp, th = find_nominal_point(np.asarray(nomianl_points[(state_index,a)]))
                        nominal_threshold[(state_index,a)] = (trp, th)
                    
                    threshold[0].append(state_index)
                    threshold[1].append(a)
                    threshold[2].append(th)
                    
                    trp /= np.sum(trp)
                    
                    for s_index, s_next in enumerate(next_states):
                        rmdp.add_transition(state_index, a, s_next, trp[s_index], get_reward(s_next, self.resolution, self.grid_x, self.grid_y))
                    
                    #Add the current transition to the RMDP
                    #for next_st in range():
                    #    rmdp.add_transition(s, a, next_st, trp[int(next_st)], rewards[next_st])
            
            #Solve the current RMDP
            rsol = rmdp.rsolve_mpi(b"optimistic_l1",threshold)
            
            #If the whole MDP is unchanged, meaning the new value function didn't change the uncertanty
            #set for any state-action, no need to iterate more!
            if is_mdp_unchanged or i==num_update-1:
                #print("**** Add Values *****")
                #print("MDP remains unchanged after number of iteration:",i)
                #print("rsol.valuefunction",rsol.valuefunction)
                return rsol
            
            valuefunction = rsol.valuefunction
            valuefunctions.append(valuefunction)
            i+=1
            #except Exception as e:
            #    print("!!! Unexpected Error in RSVF !!!", sys.exc_info()[0])
            #    print(e)
            #    continue
            
        #return under_estimate, real_regret, violation

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
    
    ofvf_learn = OFVF(env, resolution, num_runs, num_episodes, horizon)
    
    res = ofvf_learn.train(num_bayes_samples, 0.0)
    print(res[4])

### save the result
if __name__ == "__main__":
    with open('dumped_results/mountain_car_PSRL_Policy-'+date_time,'wb') as fp:
        pickle.dump(res[4].policy, fp)
        
    with open('dumped_results/mountain_car_PSRL_ValueFunction-'+date_time,'wb') as fp:
        pickle.dump(res[4].policy, fp)
        
### load a specific result file
if __name__ == "__main__":
    f = open('dumped_results/mountain_car_policy_PSRL_parametric_prior-2018-11-15 10:01:12.820398', 'rb')
    res_check = pickle.load(f)
    print(res_check)
    
### Prepare data from learned policy-valuefunction for plotting
if __name__ == "__main__":
    position, velocity, policy = [], [], []
    
    for i in ofvf_learn.grid_x:
        for j in ofvf_learn.grid_y:
            p,v = obs_to_index((i,j), ofvf_learn.env_low, ofvf_learn.env_dx)
            position.append(i)
            velocity.append(j)
            act = res[4].policy[index_to_single_index(p,v, resolution)]
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


