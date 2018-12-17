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
    
### The problem setup is simple: we've 1 non terminal state, indexed as 0. From state 0, we've 3 possible actions, each of which can lead to 3 next terminal states.
if __name__ == "__main__":
    num_states = 1
    confidence = 0.9
    num_next_states = 3
    num_actions = 3
    discount_factor = 0.9
    num_episodes = 200
    num_runs = 30
    horizon = 10
    
    # rewards for 3 possible next terminal states
    rewards = np.arange(1, num_next_states+1, dtype=float)*10
    
    # 3 possible actions, each action can lead to 3 terminal states with different probabilities. transitions[i] is the transition probability for action i.
    transitions = np.array([[0.6,0.2,0.2],[0.2,0.6,0.2],[0.2,0.2,0.6]])
    
    # Construct the true MDP with true parameters
    true_mdp = crobust.MDP(0, discount_factor)
    for a in range(num_actions):
        for s in range(num_next_states):
            true_mdp.add_transition(0, a, s+1, transitions[a, s], rewards[s])
    true_solution = true_mdp.solve_mpi()
    print(true_solution)


### UCRL2
if __name__ == "__main__":
    # ***start UCRL2***
    t=1
    regret_ucrl = np.zeros( (num_runs, num_episodes) )
    
    for m in range(num_runs):
        
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
                
                #transition_prob = crobust.worstcase_l1_dst(-rewards, p_hat[a], psi_a[a])[0]
                
                for s in range(num_next_states):
                    
                    estimated_mdp.add_transition(0, a, s+1, p_hat[a,s], rewards[a])#r_hat[a]+psi_r[a]) # as the reward is upper bounded by psi_r from mean reward r_hat
                    
                    # construct the threshold for each state-action
                    thresholds[0].append(0) # from state
                    thresholds[1].append(a) # action
                    thresholds[2].append(psi_a[a]) # allowed deviation
            #print(estimated_mdp.to_json())
        
            computed_solution = estimated_mdp.rsolve_mpi(b"optimistic_l1",np.array(thresholds)) # solve_mpi()
            computed_policy = computed_solution.policy
            #print("computed_solution", computed_solution[0][0], "true_solution[0][0]", true_solution[0][0])
            regret_ucrl[m,k] = abs(abs(computed_solution[0][0])-true_solution[0][0])
            #print("UCRL cur_solution",computed_solution, "Regret: ", regret_ucrl[m,k])
            # ***Execute policy
            Nk_ = np.zeros(num_actions)
            action = computed_policy[0] # action for the nonterminal state 0
            for _ in range(horizon): #Vk[action] < max(1,Nk[action]):
                next_state = np.random.choice(num_next_states, 1, p=transitions[action])
                reward = rewards[next_state]
                Vk[action] += 1
                t += 1
                
                Rk[action] += 1
                Pk[action,next_state] += 1
                Nk_[action] += 1

    regret_ucrl = np.mean(regret_ucrl, axis=0)
    
    print("computed_solution", computed_solution)
    # Plot regret
    plt.plot(np.cumsum(regret_ucrl))
    plt.show()
    
    
### Posterior Sampling Reinforcement Learning (PSRL)
if __name__ == "__main__":    
    
    regret_psrl = np.zeros( (num_runs, num_episodes) )
    
    for m in range(num_runs):
        # Initialize uniform Dirichlet prior
        prior = [np.ones(num_next_states) for _ in range(num_actions)]
        samples = np.zeros((num_actions, num_next_states))
        posterior = prior + samples
        # Run episodes for the PSRL
        for k in range(num_episodes):
            # Compute posterior
            posterior = posterior+samples
            samples = np.zeros((num_actions, num_next_states))
            
            sampled_mdp = crobust.MDP(0, discount_factor)
            
            for a in range(num_actions):
                trp =  np.random.dirichlet(posterior[a], 1)[0]
                for s in range(num_next_states):
                    sampled_mdp.add_transition(0, a, s+1, trp[s], rewards[s])
            
            # Compute current solution
            cur_solution = sampled_mdp.solve_mpi()
            cur_policy = cur_solution.policy
            action = cur_policy[0] # action for the nonterminal state 0
            regret_psrl[m,k] = abs(cur_solution[0][0]-true_solution[0][0])
            #print("PSRL cur_solution[0][0]",cur_solution[0][0], "Regret: ", regret_psrl[k,m])
            
            # Follow the policy to collect transition samples
            for h in range(horizon):
                next_state = np.random.choice(num_next_states, 1, p=transitions[action])
                samples[action, next_state] += 1
                
    regret_psrl = np.mean(regret_psrl, axis=0)

    plt.plot(np.cumsum(regret_psrl))
    plt.show()

    
### Bayesian UCRL
def compute_bayesian_threshold(points, nominal_point, confidence_level):
    distances = [np.linalg.norm(p - nominal_point, ord = 1) for p in points]
    confidence_rank = min(math.ceil(len(points) * confidence_level),len(points)-1)
    threshold = np.partition(distances, confidence_rank)[confidence_rank]
    return threshold
    
if __name__ == "__main__":

    num_bayes_samples = 30
    regret_bayes_ucrl = np.zeros( (num_runs, num_episodes) )
    
    for m in range(num_runs):
        # Initialize uniform Dirichlet prior
        prior = [np.ones(num_next_states) for _ in range(num_actions)]
        samples = np.zeros((num_actions, num_next_states))
        posterior = prior + samples
    
        # Run episodes for the PSRL
        for k in range(num_episodes):
            # Compute posterior
            posterior = posterior+samples
            samples = np.zeros((num_actions, num_next_states))
            confidence = 1 - 1/(k+1)

            sampled_mdp = crobust.MDP(0, discount_factor)
            
            thresholds = [[] for _ in range(3)]
            
            for a in range(num_actions):
                bayes_samples =  np.random.dirichlet(posterior[a], num_bayes_samples)
                nominal_point_bayes = np.mean(bayes_samples, axis=0)
                nominal_point_bayes /= np.sum(nominal_point_bayes)
                
                bayes_threshold = compute_bayesian_threshold(bayes_samples, nominal_point_bayes, confidence)
                
                #transition_prob = crobust.worstcase_l1_dst(-rewards, nominal_point_bayes, bayes_threshold)[0]
                
                #print("nominal_point_bayes",nominal_point_bayes,"transition_prob",transition_prob)
                
                for s in range(num_next_states):
                    sampled_mdp.add_transition(0, a, s+1, nominal_point_bayes[s], rewards[s])
                    
                # construct the threshold for each state-action
                thresholds[0].append(0) # from state
                thresholds[1].append(a) # action
                thresholds[2].append(bayes_threshold) # allowed deviation
            
            # Compute current solution
            cur_solution = sampled_mdp.rsolve_mpi(b"optimistic_l1",np.array(thresholds)) # solve_mpi()
            cur_policy = cur_solution.policy
            action = cur_policy[0] # action for the nonterminal state 0
            regret_bayes_ucrl[m,k] = abs(abs(cur_solution[0][0])-true_solution[0][0])
            #print("Bayes UCRL cur_solution[0][0]",cur_solution[0][0], "Regret: ", regret_bayes_ucrl[k,m])


            # Follow the policy to collect transition samples
            for h in range(horizon):
                next_state = np.random.choice(num_next_states, 1, p=transitions[action])
                samples[action, next_state] += 1

    regret_bayes_ucrl = np.mean(regret_bayes_ucrl, axis=0)

    plt.plot(np.cumsum(regret_bayes_ucrl))
    plt.show()


### implement RSVF style algorithm OFVF
if __name__ == "__main__":   
    num_bayes_samples = 20
    num_update = 5
    num_next_states = 4
    rewards_ = np.array([0.0] + list(rewards))
    transitions = np.array([[0.0,0.6,0.2,0.2],[0.0,0.2,0.6,0.2],[0.0,0.2,0.2,0.6]])
    
    regret_OFVF = np.zeros( (num_runs, num_episodes) )
    
    sa_confidence = confidence # !!! Apply union bound to compute confidence for each state-action
    
    for m in range(num_runs):
        # Initialize uniform Dirichlet prior
        prior = [np.ones(num_next_states) for _ in range(num_actions)]
        samples = np.zeros((num_actions, num_next_states))
        posterior = prior + samples
    
        # Run episodes for the PSRL
        for k in range(num_episodes):
            sampled_mdp = crobust.MDP(0, discount_factor)
            
            # Compute posterior
            posterior = posterior+samples
            thresholds = [[] for _ in range(3)]
            
            #sa_confidence = confidence = 1 - 1/(k+1)
            
            posterior_transition_points = []
            for a in range(num_actions):
                bayes_samples = np.random.dirichlet(posterior[a], num_bayes_samples)
                posterior_transition_points.append(bayes_samples)
                
                nominal_point_bayes = np.mean(bayes_samples, axis=0)
                nominal_point_bayes /= np.sum(nominal_point_bayes)
                
                bayes_threshold = compute_bayesian_threshold(bayes_samples, nominal_point_bayes, confidence)
 
                for s in range(num_next_states):
                    sampled_mdp.add_transition(0, a, s, nominal_point_bayes[s], rewards_[s])
                    
                # construct the threshold for each state-action
                thresholds[0].append(0) # from state
                thresholds[1].append(a) # action
                thresholds[2].append(bayes_threshold) # allowed deviation
            
            # Compute current solution
            cur_solution = sampled_mdp.rsolve_mpi(b"optimistic_l1",np.array(thresholds)) # solve_mpi()
            cur_policy = cur_solution.policy
            action = cur_policy[0] # action for the nonterminal state 0
            regret_OFVF[m,k] = OFVF(cur_solution[0], posterior_transition_points, num_update, sa_confidence, true_solution)
            #abs(abs(cur_solution[0][0])-true_solution[0][0])
            #print("OFVF cur_solution[0][0]",cur_solution[0][0], "Regret: ", regret_OFVF[m,k])
            samples = np.zeros((num_actions, num_next_states))

            # Follow the policy to collect transition samples
            for h in range(horizon):
                next_state = np.random.choice(num_next_states, 1, p=transitions[action])
                samples[action, next_state] += 1

    regret_OFVF = np.mean(regret_OFVF, axis=0)

    plt.plot(np.cumsum(regret_OFVF))
    plt.show()
    
###
    # Plot regret
    plt.plot(np.cumsum(regret_psrl), label="PSRL", color='b', linestyle='--')
    #plt.plot(np.cumsum(regret_ucrl), label="UCRL", color='g', linestyle=':')
    plt.plot(np.cumsum(regret_bayes_ucrl), label="Bayes UCRL", color='g', linestyle=':')
    plt.plot(np.cumsum(regret_OFVF), label="OFVF", color='r', linestyle='-.')
    plt.legend(loc='best', fancybox=True, framealpha=0.3)
    plt.xlabel("num_episodes")
    plt.ylabel("cumulative regret")
    plt.title("Worst Case Regret for Single State problem")
    plt.grid()
    plt.show()


###
    plt.plot(np.cumsum(worst_regret_psrl), label="PSRL", color='b', linestyle=':')
    #plt.plot(np.cumsum(worst_regret_ucrl), label="UCRL", color='c', linestyle=':')
    plt.plot(np.cumsum(worst_regret_bayes_ucrl), label="Bayes UCRL", color='g', linestyle='--')
    plt.plot(np.cumsum(worst_regret_ofvf), label="OFVF", color='r', linestyle='-.')
    plt.legend(loc='best', fancybox=True, framealpha=0.3)
    plt.xlabel("num_episodes")
    plt.ylabel("cumulative regret")
    plt.title("Worst Case Regret for RiverSwim Problem")
    plt.grid()
    plt.show()



###
def OFVF(valuefunctions, posterior_transition_points, num_update, sa_confidence, true_solution):
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
    s = 0
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
        rmdp = crobust.MDP(0, discount_factor)

        for a in range(num_actions):
            
            bayes_points = np.asarray(posterior_transition_points[a])

            RSVF_nomianlPoints = []
            
            #for bayes_points in trans:
            #print("bayes_points", bayes_points, "valuefunctions[-1]", valuefunctions[-1])
            ivf = construct_uset_known_value_function(bayes_points, valuefunctions[-1], sa_confidence)
            RSVF_nomianlPoints.append(ivf[2])
            new_trp = np.mean(RSVF_nomianlPoints, axis=0)
            
            if (s,a) not in nomianl_points:
                nomianl_points[(s,a)] = []
            
            trp, th = None, 0
            #If there's a previously constructed L1 ball. Check whether the new nominal point
            #resides outside of the current L1 ball & needs to be considered.
            if (s,a) in nominal_threshold:
                old_trp, old_th = nominal_threshold[(s,a)][0], nominal_threshold[(s,a)][1]
                
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
                nomianl_points[(s,a)].append(new_trp)
                
                #Find the center of the L1 ball for the nominal points with different 
                #value functions
                trp, th = find_nominal_point(np.asarray(nomianl_points[(s,a)]))
                nominal_threshold[(s,a)] = (trp, th)
            
            threshold[0].append(s)
            threshold[1].append(a)
            threshold[2].append(th)
            
            trp /= np.sum(trp)
            #Add the current transition to the RMDP
            for next_st in range(num_next_states):
                rmdp.add_transition(s, a, next_st, trp[int(next_st)], rewards_[next_st])
        
        #print("rmdp", rmdp.to_json())
        
        #Solve the current RMDP
        rsol = rmdp.rsolve_mpi(b"optimistic_l1",threshold)
        
        violation = 0
        
        #If the whole MDP is unchanged, meaning the new value function didn't change the uncertanty
        #set for any state-action, no need to iterate more!
        if is_mdp_unchanged or i==num_update-1:
            print("**** Add Values *****")
            print("MDP remains unchanged after number of iteration:",i)
            print("rsol.valuefunction",rsol.valuefunction)  
                            
            return abs(rsol[0][0] - true_solution[0][0])
        
        valuefunction = rsol.valuefunction
        valuefunctions.append(valuefunction)
        i+=1
        #except Exception as e:
        #    print("!!! Unexpected Error in RSVF !!!", sys.exc_info()[0])
        #    print(e)
        #    continue
        
    #return under_estimate, real_regret, violation






















