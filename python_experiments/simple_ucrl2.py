import random
import math
import numpy as np
from scipy.stats import norm
from craam import crobust
from scipy import stats
#import tqdm
import pickle
import datetime

### The problem setup is simple: we've 1 non terminal state, indexed as 0. From state 0, we've 3 possible actions, each of which can lead to 3 next terminal states.
if __name__ == "__main__":
    confidence = 0.95
    num_next_states = 3
    num_actions = 3
    discount_factor = 0.9
    
    # rewards for 3 possible next terminal states
    rewards = -np.arange(1, num_next_states+1, dtype=float)*10
    
    # 3 possible actions, each action can lead to 3 terminal states with different probabilities. transitions[i] is the transition probability for action i.
    transitions = np.array([[0.6,0.2,0.2],[0.2,0.6,0.2],[0.2,0.2,0.6]])
    
    # ***start UCRL2***
    num_episodes = 30
    t=1
    
    # state-action counts
    Nk = np.zeros(num_actions)
    Nk_ = np.zeros(num_actions)
    
    # accumulated rewards
    Rk = np.zeros(num_actions)
    
    # accumulated transition counts, initialized to uniform transition
    Pk = np.ones( (num_actions, num_next_states) )/num_next_states
    
    for k in range(num_episodes):
        
        # ***Initialize
        tk = t #set the start time of episode k
        Vk = np.zeros(num_actions) # init the state-action count for episode k
        Nk += Nk_
        r_hat = [ Rk[a]/max(1,Nk[a]) for a in range(num_actions)]
        p_hat = np.array([ Pk[a,s]/max(1,Nk[a]) for a in range(num_actions) for s in range(num_next_states) ]).reshape((num_actions,num_next_states))
        
        
        # ***Compute policy
        psi_r = [math.sqrt(7*math.log(2*num_next_states*num_actions*tk/confidence)/(2*max(1,Nk[a]))) for a in range(num_actions)]
        psi_a = [math.sqrt(14*num_next_states*math.log(2*num_actions*tk/confidence)/(max(1,Nk[a]))) for a in range(num_actions)]
        
        estimated_mdp = crobust.MDP(0, discount_factor)
        thresholds = [[] for _ in range(3)]
    
        for a in range(num_actions):
            for s in range(num_next_states):
                estimated_mdp.add_transition(0, a, s+1, p_hat[a,s], r_hat[a]+psi_r[a]) # as the reward is upper bounded by psi_r from mean reward r_hat
                
                # construct the threshold for each state-action
                thresholds[0].append(0) # from state
                thresholds[1].append(a) # action
                thresholds[2].append(psi_a[a]) # allowed deviation
        #print(estimated_mdp.to_json())
    
        computed_solution = estimated_mdp.rsolve_mpi(b"robust_l1",np.array(thresholds))
        computed_policy = computed_solution.policy
        print("episode", k , "computed_solution", computed_solution, "computed_policy", computed_policy)
        
        
        # ***Execute policy
        Nk_ = np.zeros(num_actions)
        action = computed_policy[0] # action for the nonterminal state 0
        while Vk[action] < max(1,Nk[action]):
            next_state = np.random.choice(num_next_states, 1, p=transitions[action])
            reward = rewards[next_state]
            Vk[action] += 1
            t += 1
            
            Rk[action] += 1
            Pk[action,next_state] += 1
            Nk_[action] += 1

### To check & see the solution if the MDP is solved normally, without being optimistic
if __name__ == "__main__":
    print("Normal MDP Solution")
    orig_sol = estimated_mdp.solve_mpi()
    orig_policy = orig_sol.policy
    
    print("orig_sol", orig_sol, "orig_policy", orig_policy)
    
