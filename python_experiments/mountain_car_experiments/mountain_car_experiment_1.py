import gym
from gym import wrappers
from mountainCar_BayesUCRL import *
from mountainCar_PSRL import *
from mountainCar_QLearning import *
from mountainCar_OFVF import *

# cd /home/reazul/PhD_Research/Bayesian_Exploration/Codes/Bayes_explore/bayesian_exploration/python_experiments/mountain_car_experiments
if __name__ == "__main__": 
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
            
    discount_factor = 1.0
    
    # discretization resolution
    resolution = 30
    
    # number of times to run the experiment
    num_runs = 10
    
    # number of episodes for each run
    num_episodes = 100
    
    # horizon for policy execution to collect samples for the next episode
    horizon = 100

    print("train with Q-learning")
    max_episodes, max_iterations = 200000, 20000 #00
    q_learn = QLearning(env, resolution, max_episodes, max_iterations)
    q_init_ret, _, _ = q_learn.train_qlearning()
    
    print("train with BayesUCRL")
    # number of bayes samples for BayesUCRL
    num_bayes_samples = 100
    bucrl_learn = BayesUCRL(env, resolution, num_runs, num_episodes, horizon)
    bucrl_res = bucrl_learn.train(num_bayes_samples, q_init_ret)

    print("train with PSRL")    
    psrl_learn = PSRL(env, resolution, num_runs, num_episodes, horizon)    
    psrl_res = psrl_learn.train(q_init_ret)
    
    print("train with ofvf")    
    ofvf_learn = OFVF(env, resolution, num_runs, num_episodes, horizon)
    ofvf_res = ofvf_learn.train(num_bayes_samples, q_init_ret) #psrl_learn.train(q_init_ret)
    
###
if __name__ == "__main__":
    #plt.plot(np.cumsum(psrl_res[0]), label="PSRL", color='b', linestyle=':')
    #plt.plot(np.cumsum(worst_regret_ucrl), label="UCRL", color='c', linestyle=':')
    plt.plot(np.cumsum(bucrl_res[0]), label="Bayes UCRL", color='g', linestyle='--')
    plt.plot(np.cumsum(ofvf_res[0]), label="OFVF", color='r', linestyle='-.')
    plt.legend(loc='best', fancybox=True, framealpha=0.3)
    plt.xlabel("num_episodes")
    plt.ylabel("cumulative regret")
    plt.title("Worst Case Regret for RiverSwim Problem")
    plt.grid()
    plt.show()
    
###
doubles1 = [2 * n for n in range(50)]

doubles2 = list(2 * n for n in range(50))

print(doubles1)
print(doubles2)


   
   
   
   
   