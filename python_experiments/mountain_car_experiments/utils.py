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

date_time = str(datetime.datetime.now())



def obs_to_index(obs, env_low, env_dx):
    """ 
    Maps an observation to discrete state index for both position and velocity 
    
    Parameters
    -----------
    obs: The observation tuple
    
    Returns
    --------
    a: discritized position index, 
    b: discritized velocity index
    """
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b

def index_to_obs(a,b, grid_x, grid_y):
    """ 
    Maps position and velocity indices to observation 
    
    Parameters
    -----------
    a: position index
    b: velocity index
    
    Returns
    --------
    position: position value in the continuous state space
    velocity: velocity value in the continuous state space
    """
    position = grid_x[a]
    velocity = grid_y[b]
    return position, velocity
    
#print(index_to_obs(3,0))

def index_to_single_index(a,b, resolution):
    """ 
    Calculates a single unique index based on the position and velocity
    
    Parameters
    -----------
    a: position index
    b: velocity index
    
    Returns
    --------
    single unique index
    """
    return a*resolution+b

def single_index_to_index(x, resolution):
    """ 
    computes the position and velocity indices from the single unique index
    
    Parameters
    -----------
    x: unique index
    
    Returns
    --------
    position and velocity indices
    """
    return x//resolution, x%resolution

#print(index_to_single_index(2,3))
#print(single_index_to_index(23))

def set_parametric_prior(resolution, p_index, v_index, action, max_prior):
    """
    Computes a parametric Dirichlet prior distribution based on current state-action values
    
    Parameters
    -----------
    p_index: position index
    v_index: velocity index
    action: action to take
    
    Returns
    --------
    prior: Dirichlet distribution for the prior
    """
    # define the magnitude of action effects over the bounded region. e.g. if action 'left' or '0' is chosen, the probability for
    # the car to be in a state in the left is higher. This simple heuristic is applied on the prior distribution.
    effect_left = -3 if action==0 else -2 if action==1 else -1
    effect_right = 1 if action==0 else 2 if action==1 else 3
    
    # compute the bounding box for the position and velocity values on the discritized state space
    p_min = max(p_index+effect_left, 0)
    p_max = min(p_index+effect_right, resolution-1)
    
    v_min = max(v_index+effect_left, 0)
    v_max = min(v_index+effect_right, resolution-1)
    
    # assign Dirichlet conjugate values, proportional to the nearby likelihood heuristic: the car is more likely to be in a nearby 
    # state of the current state.
    prior = {}
    for p in range(p_min, p_max+1):
        for v in range(v_min, v_max+1):
            normalizer = abs(p-p_index)+abs(v-v_index)+1
            prior[(p,v)] = 1 #max_prior//normalizer

    return prior

#print(set_parametric_prior(5, 4, 1,1,1))

def obtain_parametric_priors(resolution, num_actions):
    """
    Computes a parametric Dirichlet prior distributions for all state action pairs
    
    Parameters
    -----------
    
    Returns
    --------
    priors: prior Dirichlet distribution for all state-action
    """
    # maximum prior magnitude for any discritized state
    max_prior = 10
    
    priors = []
    
    for p in range(resolution):
        for v in range(resolution):
            for a in range(num_actions):
                priors.append(set_parametric_prior(resolution, p, v, a, max_prior))
    
    priors = np.array(priors).reshape(resolution,resolution,num_actions)
    #print("priors", priors[5,5,0])
    return priors

def get_reward(state, resolution, grid_x, grid_y):
    """
    Reward is -1 until the goal position 0.5 is reached, velocity doesn't matter in computing the reward.
    
    Parameters
    -----------
    state: single value representing the discretized state
    
    return: reward for the corresponding state
    """
    a,b = single_index_to_index(state, resolution)
    position = index_to_obs(a, b, grid_x, grid_y )[0]
    if position >= 0.5:
        return 0
    return -1
#get_reward(25)

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
    

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

#array = np.array((2,-2))
#print(totuple(array))


