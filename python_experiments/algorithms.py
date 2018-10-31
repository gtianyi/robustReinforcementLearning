import random
import math
import numpy as np
from gurobipy import *
import sys
from enum import Enum
from scipy import stats
from scipy.stats import norm
from craam import crobust


def PSRL(num_states, num_actions, num_next_states, true_transitions, rewards, discount_factor, num_episodes, num_runs, true_solution):
    """
    Implements the Posterior Sampling RL algorithm described in Osband2013 ((More) Efficient Reinforcement Learning via Posterior Sampling) paper.
    
    Parameters
    ----------
    num_states : int
        Number of states in the MDP
    num_actions : int
        Number of actions in the MDP
    num_next_states : int
        Number of possible next states. A common formulation is to keep it same as num_states, but with transition probabilities adjusted accordingly.
    true_transitions : numpy array
        num_states x num_actions x num_next_states dimensional array containing tru transition parameters.
    rewards : numpy array
        num_states dimensional array containing rewards for each state
    discount_factor : float
        Discount factor for the MDP
    num_episodes : int
        Number of episodes to run
    num_runs : int
        Number of runs in each episode, to take average of
    true_solution : Solution object of CRAAM
        The solution of the true MDP
        
    Returns
    --------
    numpy array
        Computed regret
    """
    
    regret_psrl = np.zeros( (num_runs, num_episodes) )
    
    for m in range(num_runs):
        # Initialize uniform Dirichlet prior
        prior = np.ones( (num_states, num_actions, num_next_states) )    
        samples = np.zeros( (num_states, num_actions, num_next_states) )
        posterior = prior + samples
        # Run episodes for the PSRL
        for k in range(num_episodes):
            sampled_mdp = crobust.MDP(0, discount_factor)
            
            # Compute posterior
            posterior = posterior+samples
            
            for s in range(num_states):
                for a in range(num_actions):
                    trp =  np.random.dirichlet(posterior[s,a], 1)[0]
                    for s_next in range(num_next_states):
                        sampled_mdp.add_transition(s, a, s_next, trp[s_next], rewards[s_next])
    
            # Compute current solution
            cur_solution = sampled_mdp.solve_mpi()
            cur_policy = cur_solution.policy

            #action = cur_policy[0] # action for the nonterminal state 0
            regret_psrl[m,k] = abs(cur_solution[0][0]-true_solution[0][0])
            #print("PSRL cur_solution[0][0]",cur_solution[0][0], "Regret: ", regret_psrl[k,m])
            samples = np.zeros((num_states, num_actions, num_next_states))
            
            # Follow the policy to collect transition samples
            cur_state = 0
            for h in range(horizon):
                action = cur_policy[cur_state]
                #print("cur_state", cur_state, "cur_action", action)
                next_state = np.random.choice(num_next_states, 1, p=true_transitions[cur_state, action])[0]
                #print("next_state", next_state)
                samples[cur_state, action, next_state] += 1
                cur_state = next_state
                
    #regret_psrl = np.mean(regret_psrl, axis=1)
    return np.amin(regret_psrl, axis=0), np.mean(regret_psrl, axis=0)
    
def compute_bayesian_threshold(points, nominal_point, confidence_level):
    """
    Computes an empirical L1 threshold from samples from a posterior distribution
    
    Parameters
    ----------
    points : numpy array
        Array of posterior points
    nominal_point : numpy array
        Array containing nominal/average transition probabilities
    confidence_level : float
        Required confidence level
        
    Returns
    -------
    float
        threshold value
    """
    distances = [np.linalg.norm(p - nominal_point, ord = 1) for p in points]
    confidence_rank = min(math.ceil(len(points) * confidence_level),len(points)-1)
    #print(confidence_level, confidence_rank)
    threshold = np.partition(distances, confidence_rank)[confidence_rank]
    return threshold
    
def BayesUCRL(num_states, num_actions, num_next_states, true_transitions, rewards, discount_factor, confidence, num_bayes_samples, num_episodes, num_runs, true_solution):
    """
    Implements the Bayes UCRL idea. Computes ambiguity set from posterior samples for required confidence levels.
    
    Parameters
    ----------
    num_states : int
        Number of states in the MDP
    num_actions : int
        Number of actions in the MDP
    num_next_states : int
        Number of possible next states. A common formulation is to keep it same as num_states, but with transition probabilities adjusted accordingly.
    true_transitions : numpy array
        num_states x num_actions x num_next_states dimensional array containing tru transition parameters.
    rewards : numpy array
        num_states dimensional array containing rewards for each state
    discount_factor : float
        Discount factor for the MDP
    confidence : float
        The required PAC confidence
    num_bayes_samples : int
        Number of Bayes samples to be taken from posterior
    num_episodes : int
        Number of episodes to run
    num_runs : int
        Number of runs in each episode, to take average of
    true_solution : Solution object of CRAAM
        The solution of the true MDP
        
    Returns
    --------
    numpy array
        Computed regret
    """  
    #num_bayes_samples = 20
    
    regret_bayes_ucrl = np.zeros( (num_runs, num_episodes) )
    
    for m in range(num_runs):
        # Initialize uniform Dirichlet prior
        prior = np.ones( (num_states, num_actions, num_next_states) )    
        samples = np.zeros((num_states, num_actions, num_next_states))
        posterior = prior + samples
        # Run episodes for the PSRL
        for k in range(num_episodes):
            sampled_mdp = crobust.MDP(0, discount_factor)
            
            # Compute posterior
            posterior = posterior+samples
            thresholds = [[] for _ in range(3)]
            
            confidence = 1-1/(k+1)
            sa_confidence = 1-(1-confidence)/(num_states*num_actions) # !!! Apply union bound to compute confidence for each state-action
            
            for s in range(num_states):
                for a in range(num_actions):
                    bayes_samples =  np.random.dirichlet(posterior[s,a], num_bayes_samples)
                    nominal_point_bayes = np.mean(bayes_samples, axis=0)
                    nominal_point_bayes /= np.sum(nominal_point_bayes)
                    
                    bayes_threshold = compute_bayesian_threshold(bayes_samples, nominal_point_bayes, sa_confidence)
                    
                    for s_next in range(num_next_states):
                        sampled_mdp.add_transition(s, a, s_next, nominal_point_bayes[s_next], rewards[s_next])
                        
                    # construct the threshold for each state-action
                    thresholds[0].append(s) # from state
                    thresholds[1].append(a) # action
                    thresholds[2].append(bayes_threshold) # allowed deviation
            
            # Compute current solution
            cur_solution = sampled_mdp.rsolve_mpi(b"optimistic_l1",np.array(thresholds))
            cur_policy = cur_solution.policy
            #action = cur_policy[0] # action for the nonterminal state 0
            regret_bayes_ucrl[m,k] = abs(abs(cur_solution[0][0])-true_solution[0][0])
            #print("Bayes UCRL cur_solution[0][0]",cur_solution[0][0], "Regret: ", regret_bayes_ucrl[k,m])
            samples = np.zeros((num_states, num_actions, num_next_states))
                
            # Follow the policy to collect transition samples
            cur_state = 0
            for h in range(horizon):
                action = cur_policy[cur_state]
                #print("cur_state", cur_state, "cur_action", action)
                next_state = np.random.choice(num_next_states, 1, p=true_transitions[cur_state, action])[0]
                #print("next_state", next_state)
                samples[cur_state, action, next_state] += 1
                cur_state = next_state

    #regret_bayes_ucrl = np.mean(regret_bayes_ucrl, axis=1)
    return np.amin(regret_bayes_ucrl, axis=0), np.mean(regret_bayes_ucrl, axis=0)
    
    
def UCRL2(num_states, num_actions, num_next_states, true_transitions, rewards, discount_factor, num_episodes, num_runs, true_solution):
    """
    Implements the UCRL2 algorithm described in Jaksch2010 (Near-optimal Regret Bounds for Reinforcement Learning) paper.
    
    Parameters
    ----------
    num_states : int
        Number of states in the MDP
    num_actions : int
        Number of actions in the MDP
    num_next_states : int
        Number of possible next states. A common formulation is to keep it same as num_states, but with transition probabilities adjusted accordingly.
    true_transitions : numpy array
        num_states x num_actions x num_next_states dimensional array containing tru transition parameters.
    rewards : numpy array
        num_states dimensional array containing rewards for each state
    discount_factor : float
        Discount factor for the MDP
    num_episodes : int
        Number of episodes to run
    num_runs : int
        Number of runs in each episode, to take average of
    true_solution : Solution object of CRAAM
        The solution of the true MDP
        
    Returns
    --------
    numpy array
        Computed regret
    """
    # ***start UCRL2***
    #num_next_states = num_states
    regret_ucrl = np.zeros( (num_runs, num_episodes) )
    
    #num_runs, num_episodes = 1, 1
    
    for m in range(num_runs):
        t=1
        # state-action counts
        Nk = np.zeros( (num_states, num_actions) )
        Nk_ = np.zeros( (num_states, num_actions) )
        
        # accumulated rewards
        Rk = np.zeros( (num_states, num_actions) )
        
        # accumulated transition counts, initialized to uniform transition
        Pk = np.ones( (num_states, num_actions, num_next_states) )/num_next_states
    
        for k in range(num_episodes):

            # ***Initialize
            tk = t #set the start time of episode k
            Vk = np.zeros( (num_states, num_actions) ) # init the state-action count for episode k
            Nk += Nk_
            r_hat = [ Rk[s,a]/max(1,Nk[s,a]) for s in range(num_states) for a in range(num_actions)]
            p_hat = np.array([ Pk[s,a,s_next]/max(1,Nk[s,a]) for s in range(num_states) for a in range(num_actions)\
                                                                for s_next in range(num_next_states) ]).reshape((num_states, num_actions,num_next_states))
            for s in range(num_states):
                for a in range(num_actions):
                    p_hat[s,a] /= np.sum(p_hat[s,a])
                    
            # ***Compute policy
            psi_r = [math.sqrt(7*math.log(2*num_next_states*num_actions*tk/confidence)/(2*max(1,Nk[s,a]))) for s in range(num_states) for a in range(num_actions)]
            psi_a = [math.sqrt(14*num_next_states*math.log(2*num_actions*tk/confidence)/(max(1,Nk[s,a]))) for s in range(num_states) for a in range(num_actions)]
            
            estimated_mdp = crobust.MDP(0, discount_factor)
            thresholds = [[] for _ in range(3)]
            
            for s in range(num_states):
                for a in range(num_actions):
                    
                    for s_next in range(num_next_states):
                        estimated_mdp.add_transition(s, a, s_next, p_hat[s,a,s_next], rewards[s_next]) # as the reward is upper bounded by psi_r from mean reward r_hat
                    # construct the threshold for each state-action
                    thresholds[0].append(s) # from state
                    thresholds[1].append(a) # action
                    thresholds[2].append(psi_a[a]) # allowed deviation
            #print(estimated_mdp.to_json())

            computed_solution = estimated_mdp.rsolve_mpi(b"optimistic_l1",np.array(thresholds))
            computed_policy = computed_solution.policy

            regret_ucrl[m,k] = abs(abs(computed_solution[0][0])-true_solution[0][0])
        
            # ***Execute policy
            Nk_ = np.zeros( (num_states, num_actions) )
            cur_state = 0

            for h in range(horizon): #Vk[action] < max(1,Nk[action]):
                action = computed_policy[cur_state]
                next_state = np.random.choice(num_next_states, 1, p=true_transitions[cur_state, action])[0]
                reward = rewards[next_state]
                Vk[cur_state, action] += 1
                t += 1
                
                Rk[cur_state, action] += 1
                Pk[cur_state, action, next_state] += 1
                Nk_[cur_state, action] += 1
                cur_state = next_state

    #regret_ucrl = np.mean(regret_ucrl, axis=0)
    return np.amin(regret_ucrl, axis=0), np.mean(regret_ucrl, axis=0)
    


def construct_uset_known_value_function(transition_points, value_function, confidence):
    """
    Computes the robust return and a threshold that achieves the desired confidence level
    for a single state and action.

    @param transition_points Samples from the posterior distribution of the transition probabilities
    @param value_function Assumed optimal value function
    @param confidence Desired confidence level, such as 0.99
    """
    points = []

    for p in transition_points:
        points.append( (p,p@value_function) )
    points.sort(key=lambda x: x[1])

    conf_rank = min(math.ceil(len(transition_points)*confidence),len(transition_points)-1 )
    #print("confidence_rank", conf_rank, "len(trans_points)", len(transition_points), "confidence",confidence,"product",confidence*len(transition_points))
    robust_ret = points[-conf_rank][1]
    robust_th = 0 #np.linalg.norm(points[-int(conf_rank)][0]-points[-int(conf_rank/2)][0], ord=1)
    nominal_point = points[-conf_rank][0]
    
    return (robust_ret, robust_th, nominal_point)

def find_nominal_point(p):
    """
    Find nominal point for the uncertainty set using LP
    """
    num_p = p.shape[0]
    num_d = p.shape[1]
    m = Model("nominal")
    u = m.addVar(name="u", lb=0)
    
    y = m.addVars(range(num_p*num_d), vtype=GRB.CONTINUOUS, obj=0.0, name="y")
    # new nominal point
    beta = m.addVars(range(num_d), vtype=GRB.CONTINUOUS, obj=0.0, name="beta", lb=0)
    
    m.setObjective(u, GRB.MINIMIZE)
    
    for i in range(num_p):
        m.addConstr(u, GRB.GREATER_EQUAL, quicksum(y[i*num_d+j] for j in range(num_d)), "u_"+str(i))

    for i in range(num_p):
        for j in range(num_d):
            m.addConstr(y[i*num_d+j], GRB.GREATER_EQUAL, p[i,j]-beta[j], "c1"+str(i))
            m.addConstr(y[i*num_d+j], GRB.GREATER_EQUAL, beta[j]-p[i,j], "c2"+str(i))

    m.setParam( 'OutputFlag', False )
    m.optimize()
    
    #print('Obj: %g' % m.objVal) 
    
    #for v in m.getVars():
    #    print('%s %g' % (v.varName, v.x))
    
    threshold = 0
    for v in m.getVars():
        if v.varName == "u":
            threshold = v.x
            break
            
    nominal_params = m.getAttr('x', beta)
    
    nominal_p = []
    for i in range(num_d):
        nominal_p.append(nominal_params[i])
    
    return nominal_p, threshold#tuple(nominal_p)


def Optimism_VF(num_states, num_actions, num_next_states, true_transitions, rewards, discount_factor, confidence, num_bayes_samples, num_episodes, num_runs, true_solution):  
    #num_bayes_samples = 20
    num_update = 10
    
    confidences = []
    regret_OFVF = np.zeros( (num_runs, num_episodes) )
    violations = np.zeros( (num_runs, num_episodes) )
    
    for m in range(num_runs):
        # Initialize uniform Dirichlet prior
        prior = np.ones( (num_states, num_actions, num_next_states) )
        samples = np.zeros( (num_states, num_actions, num_next_states) )
        posterior = prior + samples
        # Run episodes for the PSRL
        for k in range(num_episodes):
            sampled_mdp = crobust.MDP(0, discount_factor)
            confidence = 1-1/(k+1)
            sa_confidence = 1-(1-confidence)/(num_states*num_actions) # !!! Apply union bound to compute confidence for each state-action
            if m==0:
                confidences.append(confidence)
            
            # Compute posterior
            posterior = posterior+samples
            thresholds = [[] for _ in range(3)]
            
            posterior_transition_points = {}
            for s in range(num_states):
                for a in range(num_actions):
                    bayes_samples = np.random.dirichlet(posterior[s,a], num_bayes_samples)
                    posterior_transition_points[(s,a)] = bayes_samples
                    
                    nominal_point_bayes = np.mean(bayes_samples, axis=0)
                    nominal_point_bayes /= np.sum(nominal_point_bayes)
                    
                    bayes_threshold = compute_bayesian_threshold(bayes_samples, nominal_point_bayes, sa_confidence)
    
                    for s_next in range(num_next_states):
                        sampled_mdp.add_transition(s, a, s_next, nominal_point_bayes[s_next], rewards[s_next])
                        
                    # construct the threshold for each state-action
                    thresholds[0].append(s) # from state
                    thresholds[1].append(a) # action
                    thresholds[2].append(bayes_threshold) # allowed deviation
            
            # Compute current solution
            cur_solution = sampled_mdp.rsolve_mpi(b"optimistic_l1",np.array(thresholds)) # solve_mpi()
            
            rsol = OFVF(num_states, num_actions, num_next_states, cur_solution[0], posterior_transition_points, num_update, sa_confidence)
            
            regret_OFVF[m,k] = abs(rsol[0][0] - true_solution[0][0])
            
            violations[m,k] = rsol[0][0] - true_solution[0][0]
            
            rpolicy = rsol.policy

            samples = np.zeros((num_states, num_actions, num_next_states))

            # Follow the policy to collect transition samples
            cur_state = 0
            for h in range(horizon):
                action = rpolicy[cur_state]
                #print("cur_state", cur_state, "cur_action", action)
                next_state = np.random.choice(num_next_states, 1, p=true_transitions[cur_state, action])[0]
                #print("next_state", next_state)
                samples[cur_state, action, next_state] += 1
                cur_state = next_state

    #regret_OFVF = np.mean(regret_OFVF, axis=1)
    #plt.plot(np.cumsum(regret_OFVF))
    #plt.show()
    
    violations = np.mean(violations<0, axis=0)
    
    return np.amin(regret_OFVF, axis=0), np.mean(regret_OFVF, axis=0), violations, confidences


def OFVF(num_states, num_actions, num_next_states, valuefunctions, posterior_transition_points, num_update, sa_confidence):
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
        rmdp = crobust.MDP(0, discount_factor)
        #print("update", i)
        for s in range(num_states):
            for a in range(num_actions):
                
                bayes_points = np.asarray(posterior_transition_points[s,a])
    
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
                    rmdp.add_transition(s, a, next_st, trp[int(next_st)], rewards[next_st])
        
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


