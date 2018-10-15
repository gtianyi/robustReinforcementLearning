
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
    # Initialize uniform Dirichlet prior
    prior = np.ones( (num_states, num_actions, num_next_states) )    
    samples = np.zeros( (num_states, num_actions, num_next_states) )
    posterior = prior + samples
    
    regret_psrl = np.zeros( (num_episodes, num_runs) )
    
    # Run episodes for the PSRL
    for k in range(num_episodes):
        for m in range(num_runs):
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
            regret_psrl[k,m] = abs(cur_solution[0][0]-true_solution[0][0])
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
    return np.mean(regret_psrl, axis=1)
    
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
    confidence_rank = math.ceil(len(points) * confidence_level)
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
    # Initialize uniform Dirichlet prior
    prior = np.ones( (num_states, num_actions, num_next_states) )    
    samples = np.zeros((num_states, num_actions, num_next_states))
    posterior = prior + samples
    
    regret_bayes_ucrl = np.zeros( (num_episodes, num_runs) )
    
    # Run episodes for the PSRL
    for k in range(num_episodes):
        for m in range(num_runs):
            sampled_mdp = crobust.MDP(0, discount_factor)
            
            # Compute posterior
            posterior = posterior+samples
            thresholds = [[] for _ in range(3)]
            
            for s in range(num_states):
                for a in range(num_actions):
                    bayes_samples =  np.random.dirichlet(posterior[s,a], num_bayes_samples)
                    nominal_point_bayes = np.mean(bayes_samples, axis=0)
                    nominal_point_bayes /= np.sum(nominal_point_bayes)
                    
                    bayes_threshold = compute_bayesian_threshold(bayes_samples, nominal_point_bayes, confidence)
                    
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
            regret_bayes_ucrl[k,m] = abs(abs(cur_solution[0][0])-true_solution[0][0])
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
    return np.mean(regret_bayes_ucrl, axis=1)
    
    
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
    return np.mean(regret_ucrl, axis=0)
    
    
