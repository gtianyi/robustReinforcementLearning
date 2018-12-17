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

print(datetime.datetime.now())
# cd /home/reazul/PhD_Research/Bayesian_Exploration/Codes/Bayes_explore/bayesian_exploration/python_experiments

"""
**** RiverSwim MDP problem, described in Osband2013 paper
@article{Osband2013,
abstract = {Most provably-efficient learning algorithms introduce optimism about poorly-understood states and actions to encourage exploration. We study an alternative approach for efficient exploration, posterior sampling for reinforcement learning (PSRL). This algorithm proceeds in repeated episodes of known duration. At the start of each episode, PSRL updates a prior distribution over Markov decision processes and takes one sample from this posterior. PSRL then follows the policy that is optimal for this sample during the episode. The algorithm is conceptually simple, computationally efficient and allows an agent to encode prior knowledge in a natural way. We establish an {\$}\backslashtilde{\{}O{\}}(\backslashtau S \backslashsqrt{\{}AT{\}}){\$} bound on the expected regret, where {\$}T{\$} is time, {\$}\backslashtau{\$} is the episode length and {\$}S{\$} and {\$}A{\$} are the cardinalities of the state and action spaces. This bound is one of the first for an algorithm not based on optimism, and close to the state of the art for any reinforcement learning algorithm. We show through simulation that PSRL significantly outperforms existing algorithms with similar regret bounds.},
author = {Osband, Ian and Russo, Daniel and {Van Roy}, Benjamin},
title = {{(More) Efficient Reinforcement Learning via Posterior Sampling}},
url = {http://arxiv.org/abs/1306.0940},
year = {2013}
}
"""
### Experiments with chained MDP
if __name__ == "__main__":
    confidence = 0.9
    num_actions = 2
    discount_factor = 0.99
    num_bayes_samples = 100
    num_episodes = 500
    num_runs = 50
    horizon = 10
    num_states = 6
    states = np.arange(num_states)
    rewards = np.zeros(num_states)
    rewards[0], rewards[num_states-1] = 5/1000, 1
    date_time = str(datetime.datetime.now())
    
    transitions = np.zeros( (num_states, num_actions, num_states) )
    
    sa_pairs = list(itertools.product(np.arange(num_states), np.arange(num_actions)))
    
    # Construct true transition matrix
    for s in range(num_states):
        # action left
        transitions[s, 0, max(s-1,0)] = 1
        
        # action right
        transitions[s, 1, s] = 0.4 if s==0 else 0.6
        transitions[s, 1, min(s+1,num_states-1)] = 0.6 if s==0 else (0.6 if s==num_states-1 else 0.35)
        transitions[s, 1, max(s-1,0)] = 0.4 if s==0 else (0.4 if s==num_states-1 else 0.05)
    
    true_mdp = crobust.MDP(0, discount_factor)
    for s in range(num_states):
        true_mdp.add_transition(s, 0, max(s-1,0), transitions[s, 0, max(s-1,0)], rewards[max(s-1,0)])
        
        true_mdp.add_transition(s, 1, s, transitions[s, 1, s], rewards[s])
        if s<num_states-1:
            true_mdp.add_transition(s, 1, min(s+1,num_states-1), transitions[s, 1, min(s+1,num_states-1)], rewards[min(s+1,num_states-1)])
        if s>0:
            true_mdp.add_transition(s, 1, max(s-1,0), transitions[s, 1, max(s-1,0)], rewards[max(s-1,0)])
        
    #print(true_mdp.to_json())
    true_solution = true_mdp.solve_mpi()
    
    with open('dumped_results/truesolution'+date_time,'wb') as fp:
        pickle.dump(true_solution.valuefunction, fp)
    
    print("true_solution", true_solution)

### Run Experiments
if __name__ == "__main__":
    #print("executing UCRL2...")
    #worst_regret_ucrl, avg_regret_ucrl = UCRL2(num_states, num_actions, num_states, transitions, rewards, discount_factor, num_episodes, num_runs, true_solution)
    print("executing PSRL...")
    worst_regret_psrl, avg_regret_psrl = PSRL(num_states, num_actions, num_states, transitions, rewards, discount_factor, num_episodes, num_runs, true_solution)
    with open('dumped_results/worst_regret_psrl'+date_time,'wb') as fp:
        pickle.dump(worst_regret_psrl, fp)
    with open('dumped_results/avg_regret_psrl'+date_time,'wb') as fp:
        pickle.dump(worst_regret_psrl, fp)
    
    
    print("executing Bayes UCRL...")
    worst_regret_bayes_ucrl, avg_regret_bayes_ucrl =  BayesUCRL(num_states, num_actions, num_states, transitions, rewards, discount_factor, confidence, num_bayes_samples, num_episodes, num_runs, true_solution)
    with open('dumped_results/worst_regret_bayes_ucrl'+date_time,'wb') as fp:
        pickle.dump(worst_regret_bayes_ucrl, fp)
    with open('dumped_results/avg_regret_bayes_ucrl'+date_time,'wb') as fp:
        pickle.dump(avg_regret_bayes_ucrl, fp)
    
    
    print("executing OFVF...")
    worst_regret_ofvf, avg_regret_ofvf,  violations, confidences = Optimism_VF(num_states, num_actions, num_states, transitions, rewards, discount_factor, confidence, num_bayes_samples, num_episodes, num_runs, true_solution)
    with open('dumped_results/worst_regret_ofvf'+date_time,'wb') as fp:
        pickle.dump(worst_regret_ofvf, fp)
    with open('dumped_results/avg_regret_ofvf'+date_time,'wb') as fp:
        pickle.dump(avg_regret_ofvf, fp)
    with open('dumped_results/violations_ofvf'+date_time,'wb') as fp:
        pickle.dump(violations, fp)
    with open('dumped_results/confidences_ofvf'+date_time,'wb') as fp:
        pickle.dump(confidences, fp)
    
### Plot worst case results
if __name__ == "__main__":
    f = open('dumped_results/worst_regret_psrl', 'rb')
    worst_regret_psrl = pickle.load(f)
    f.close()
    
    f = open('dumped_results/worst_regret_bayes_ucrl', 'rb')
    worst_regret_bayes_ucrl = pickle.load(f)
    f.close()

    f = open('dumped_results/worst_regret_ofvf', 'rb')
    worst_regret_ofvf = pickle.load(f)
    f.close()

    f = open('dumped_results/avg_regret_psrl', 'rb')
    avg_regret_psrl = pickle.load(f)
    f.close()

    f = open('dumped_results/avg_regret_bayes_ucrl', 'rb')
    avg_regret_bayes_ucrl = pickle.load(f)
    f.close()

    f = open('dumped_results/avg_regret_ofvf', 'rb')
    avg_regret_ofvf = pickle.load(f)
    f.close()

    f = open('dumped_results/violations_ofvf', 'rb')
    violations = pickle.load(f)
    f.close()

    f = open('dumped_results/confidences_ofvf', 'rb')
    confidences = pickle.load(f)
    f.close()


###
if __name__ == "__main__":
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
    
### Plot average case results
if __name__ == "__main__":
    plt.plot(np.cumsum(avg_regret_psrl), label="PSRL", color='b', linestyle=':')
    #plt.plot(np.cumsum(avg_regret_ucrl), label="UCRL", color='c', linestyle=':')
    plt.plot(np.cumsum(avg_regret_bayes_ucrl), label="Bayes UCRL", color='g', linestyle='--')
    plt.plot(np.cumsum(avg_regret_ofvf), label="OFVF", color='r', linestyle='-.')
    plt.legend(loc='best', fancybox=True, framealpha=0.3)
    plt.xlabel("num_episodes")
    plt.ylabel("cumulative regret")
    plt.title("Average Case Regret for RiverSwim Problem")
    plt.grid()
    plt.show()

### Plot violations
if __name__ == "__main__":
    plt.plot(np.array(violations), label="violations of OFVF", color='r', linestyle='-.')
    plt.plot(np.array(confidences), label="Actual confidence", color='g', linestyle='--')
    plt.legend(loc='best', fancybox=True, framealpha=0.3)
    plt.xlabel("num_episodes")
    plt.ylabel("violations")
    plt.title("Violations")
    plt.grid()
    plt.show()















    