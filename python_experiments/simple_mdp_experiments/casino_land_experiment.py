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
from algorithms import *

# cd /home/reazul/PhD_Research/Bayesian_Exploration/Codes/Bayes_explore/bayesian_exploration/python_experiments/simple_mdp_experiments

### Experiments with RiverSwim Problem
if __name__ == "__main__":
    confidence = 0.9
    discount_factor = 0.99
    num_bayes_samples = 100
    num_episodes = 100
    num_runs = 20
    horizon = 30
    
    num_states = 8
    num_actions = 4
    
    states = np.arange(num_states)
    
    transitions = np.array([
        #  s0    s1    s2    s3    s4    s5    s6    s7
        [[1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s0, 0
        [ 0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0],  # s0, 1
        [ 0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s0, 2
        [ 1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]], # s0, 3
        
        [[0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s1, 0
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0],  # s1, 1
        [ 0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s1, 2
        [ 1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]], # s1, 3
        
        [[0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s2, 0
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0],  # s2, 1
        [ 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0],  # s2, 2
        [ 0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]], # s3, 3
        
        [[0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0],  # s3, 0
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0],  # s3, 1
        [ 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0],  # s3, 2
        [ 0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0]], # s4, 3
        
        [[1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s4, 0
        [ 0.4,  0.0,  0.0,  0.0,  0.6,  0.0,  0.0,  0.0],  # s4, 1
        [ 0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0],  # s4, 2
        [ 0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0]], # s4, 3
        
        [[0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s5, 0
        [ 0.0,  0.5,  0.0,  0.0,  0.0,  0.5,  0.0,  0.0],  # s5, 1
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0],  # s5, 2
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0]], # s5, 3

        [[0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s6, 0
        [ 0.0,  0.0,  0.2,  0.0,  0.0,  0.0,  0.8,  0.0],  # s6, 1
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0],  # s6, 2
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0]], # s6, 3
        
        [[0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0],  # s7, 0
        [ 0.0,  0.0,  0.0,  0.9,  0.0,  0.0,  0.0,  0.1],  # s7, 1
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0],  # s7, 2
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0]]  # s7, 3
        ])

    rewards = np.array([
        #  s0    s1    s2    s3    s4    s5    s6    s7
        [[0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s0, 0
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s0, 1
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s0, 2
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]], # s0, 3
        
        [[0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s1, 0
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s1, 1
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s1, 2
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]], # s1, 3
        
        [[0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s2, 0
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s2, 1
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s2, 2
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]], # s2, 3
        
        [[0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s3, 0
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s3, 1
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s3, 2
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]], # s3, 3
        
        [[0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s4, 0
        [ 198,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s4, 1
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s4, 2
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]], # s4, 3
        
        [[0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s5, 0
        [ 0.0,  160,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s5, 1
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s5, 2
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]], # s5, 3

        [[0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s6, 0
        [ 0.0,  0.0,  500,  0.0,  0.0,  0.0,  0.0,  0.0],  # s6, 1
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s6, 2
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]], # s6, 3
        
        [[0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s7, 0
        [ 0.0,  0.0,  0.0,  72.,  0.0,  0.0,  0.0,  0.0],  # s7, 1
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # s7, 2
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]]  # s7, 3
        ])

    sas_pairs = list(itertools.product(np.arange(num_states), np.arange(num_actions), np.arange(num_states)))
    #print(sa_pairs)

    true_mdp = crobust.MDP(0, discount_factor)
    for sas in sas_pairs:
        s, a, s_next = sas[0], sas[1], sas[2]        
        true_mdp.add_transition(s, a, s_next, transitions[s, a, s_next], rewards[s,a,s_next])
    
    #print(true_mdp.to_json())
    true_solution = true_mdp.solve_mpi()
    
    date_time = str(datetime.datetime.now())
    with open('../dumped_results/casino_land_truesolution'+date_time,'wb') as fp:
        pickle.dump(true_solution.valuefunction, fp)
    
    print("true_solution", true_solution)

### Run Experiments
if __name__ == "__main__":
    #print("executing UCRL2...")
    #worst_regret_ucrl, avg_regret_ucrl = UCRL2(num_states, num_actions, num_states, transitions, rewards, discount_factor, num_episodes, num_runs, true_solution)
    print("executing PSRL...")
    worst_regret_psrl, avg_regret_psrl = PSRL(num_states, num_actions, num_states, transitions, rewards, discount_factor, num_episodes, num_runs, horizon, true_solution)
    with open('../dumped_results/casino_land_worst_regret_psrl'+date_time,'wb') as fp:
        pickle.dump(worst_regret_psrl, fp)
    with open('../dumped_results/casino_land_avg_regret_psrl'+date_time,'wb') as fp:
        pickle.dump(worst_regret_psrl, fp)
    
    
    print("executing Bayes UCRL...")
    worst_regret_bayes_ucrl, avg_regret_bayes_ucrl =  BayesUCRL(num_states, num_actions, num_states, transitions, rewards, discount_factor, confidence, num_bayes_samples, num_episodes, num_runs, horizon, true_solution)
    with open('../dumped_results/casino_land_worst_regret_bayes_ucrl'+date_time,'wb') as fp:
        pickle.dump(worst_regret_bayes_ucrl, fp)
    with open('../dumped_results/casino_land_avg_regret_bayes_ucrl'+date_time,'wb') as fp:
        pickle.dump(avg_regret_bayes_ucrl, fp)
    
    
    print("executing OFVF...")
    worst_regret_ofvf, avg_regret_ofvf,  violations, confidences = Optimism_VF(num_states, num_actions, num_states, transitions, rewards, discount_factor, confidence, num_bayes_samples, num_episodes, num_runs, horizon, true_solution)
    with open('../dumped_results/casino_land_worst_regret_ofvf'+date_time,'wb') as fp:
        pickle.dump(worst_regret_ofvf, fp)
    with open('../dumped_results/casino_land_avg_regret_ofvf'+date_time,'wb') as fp:
        pickle.dump(avg_regret_ofvf, fp)
    with open('../dumped_results/casino_land_violations_ofvf'+date_time,'wb') as fp:
        pickle.dump(violations, fp)
    with open('../dumped_results/casino_land_confidences_ofvf'+date_time,'wb') as fp:
        pickle.dump(confidences, fp)
    
### Plot worst case results
if __name__ == "__main__":
    f = open('../dumped_results/casino_land_worst_regret_psrl', 'rb')
    worst_regret_psrl = pickle.load(f)
    f.close()
    
    f = open('../dumped_results/casino_land_worst_regret_bayes_ucrl', 'rb')
    worst_regret_bayes_ucrl = pickle.load(f)
    f.close()

    f = open('../dumped_results/casino_land_worst_regret_ofvf', 'rb')
    worst_regret_ofvf = pickle.load(f)
    f.close()

    f = open('../dumped_results/casino_land_avg_regret_psrl', 'rb')
    avg_regret_psrl = pickle.load(f)
    f.close()

    f = open('../dumped_results/casino_land_avg_regret_bayes_ucrl', 'rb')
    avg_regret_bayes_ucrl = pickle.load(f)
    f.close()

    f = open('../dumped_results/casino_land_avg_regret_ofvf', 'rb')
    avg_regret_ofvf = pickle.load(f)
    f.close()

    f = open('../dumped_results/casino_land_violations_ofvf', 'rb')
    violations = pickle.load(f)
    f.close()

    f = open('../dumped_results/casino_land_confidences_ofvf', 'rb')
    confidences = pickle.load(f)
    f.close()


###
if __name__ == "__main__":
    plt.plot(np.cumsum(worst_regret_psrl), label="PSRL", color='b', linestyle='--')
    #plt.plot(np.cumsum(worst_regret_ucrl), label="UCRL", color='c', linestyle=':')
    plt.plot(np.cumsum(worst_regret_bayes_ucrl), label="Bayes UCRL", color='g', linestyle=':')
    plt.plot(np.cumsum(worst_regret_ofvf), label="OFVF", color='r', linestyle='-.')
    plt.legend(loc='best', fancybox=True, framealpha=0.3)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Cumulative Regret")
    plt.title("Worst Case Regret for CasinoLand Problem")
    plt.grid()
    plt.savefig("../figures/casino_land__worstcase_Bayes_PSRL_OFVF.pdf")
    plt.show()
    
### Plot average case results
if __name__ == "__main__":
    plt.plot(np.cumsum(avg_regret_psrl), label="PSRL", color='b', linestyle='--')
    #plt.plot(np.cumsum(avg_regret_ucrl), label="UCRL", color='c', linestyle=':')
    plt.plot(np.cumsum(avg_regret_bayes_ucrl), label="Bayes UCRL", color='g', linestyle=':')
    plt.plot(np.cumsum(avg_regret_ofvf), label="OFVF", color='r', linestyle='-.')
    plt.legend(loc='best', fancybox=True, framealpha=0.3)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Cumulative Regret")
    plt.title("Average Case Regret for CasinoLand Problem")
    plt.grid()
    plt.savefig("../figures/casino_land_averagecase_Bayes_PSRL_OFVF.pdf")
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














    