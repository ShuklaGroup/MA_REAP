'''
This document implements the multiagent REAP using kmeans clustering and the standardized Euclidean distance in the reward definition.
'''

import numpy as np
import matplotlib.pyplot as plt
import openmm as mm
from utils import save_pickle, load_pickle, plot_potential, run_trajectory, setup_simulation, area_explored
import os
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def clustering_MA(trajectories, n_agents, n_select, n_clusters=None, max_frames=1e5, b=1e-4, gamma=0.7, d=None):
    '''
    Clusters all (or a representative subset) of the frames in trajectories using KMeans. Returns clustering object, which will be used to select the 
    least counts clusters. The agents that created each frame are remembered and their indices are returned as well. The selected subset of the total
    frames are also returned.
    
    Args
    -------------
    trajectories (list[list[np.ndarray]]): trajectories collected so far. They should be accessed as trajectories[ith_agent][jth_trajectory].
    n_agents (int): number of agents.
    n_clusters (int), default None: number of clusters to use for KMeans. If None, a heuristic by Buenfil et al. (2021) is used to approximate the number of clusters needed.
    max_frames (int), default 1e5: maximum number of frames to use in the clustering step. If set to 0 or None, all frames are used.
    b (float), default 1e-4: coefficient for n_clusters heuristic.
    gamma (float), default 0.7: exponent for n_clusters heuristic (should theoretically be in [0.5, 1]).
    d (int), no default: intrinsic dimensionality of slow manifold for the system. Used in n_clusters heuristic. Must be defined.
   
    Returns
    -------------
    KMeans (sklearn.cluster.KMeans): fitted KMeans clustering object.
    X (np.ndarray): array of shape (max_frames, n_features) containing the subset of the data used for clustering.
    agent_idx (np.ndarray): array of shape (max_frames,) containing the index of the agent that originated each frame.
    '''
    
    # Put frames in format that is usable for KMeans
    assert(n_agents == len(trajectories))
    total_frames = 0
    trajectory = [] # All frames
    agent_index = [] # Array mapping a frame index in trajectory to its corresponding agent
    for a, agent_trajs in enumerate(trajectories):
        for traj in agent_trajs:
            total_frames += len(traj)
            trajectory.append(traj)
            agent_index.extend([a]*len(traj))
          
    trajectory = np.concatenate(trajectory)
    agent_index = np.asarray(agent_index)
    
    # Downsample number of points
    if (not max_frames) or (total_frames <= max_frames):
        X = trajectory
        agent_idx = agent_index
        
    
    elif total_frames > max_frames:
        max_frames = int(max_frames)
        rng = np.random.default_rng()
        rand_indices = rng.choice(len(trajectory), max_frames, replace=False)
        X = trajectory[rand_indices]
        agent_idx = agent_index[rand_indices]
                                  
    # Use heuristic from https://openreview.net/pdf?id=00thAjcutwh to determine number of clusters
    if (n_clusters is None):
        n_clusters = int(b*(min(total_frames, max_frames)**(gamma*d)))

    if (n_clusters < n_select):
        n_clusters = n_select # Ensure there are enough clusters to select candidates from
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, init='random').fit(X)
    
    return kmeans, X, agent_idx

def select_least_counts_MA(kmeans, X, agent_idx, n_agents, n_select=50, stakes_method="percentage", stakes_k=None):
    '''
    Select candidate clusters for new round of simulations based on least counts policy.
    
    Args
    -------------
    kmeans (sklearn.cluster.KMeans): KMeans clustering object fitted on X.
    X (np.ndarray): array of shape (n_frames, n_features). Representative subset of frames from trajectories.
    agent_idx (np.ndarray): array of shape (n_frames,) indicating which agent originated each frame.
    n_agents (int): number of agents.
    n_select (int), default 50: how many candidates to select based on least counts policy.
    stakes_method (str): one of "percentage", "max", "equal", or "logistic". 
        percentage: stakes are proportional to number of frames that agents have in the cluster.
        max: the agent with the max number of frames in the cluster has all the stakes.
        equal: all agents with at least one frame in the cluster have the same stakes.
        logistic: percentage stakes are transformed as 1/(1+e^(-k(x-0.5))) and renormalized. Must set parameter k (k >= 100 is similar to max, k <= 0.01 is similar to equal).
    stakes_k (float): parameter for logistic method for stakes calculation. Ignored if method is not set to logistic.    
    
    Returns
    -------------
    central_frames (np.ndarray): array of shape (n_select, n_features). Frames in X that are closest to the center of each candidate.
    central_frames_indices (np.ndarray): array of shape (n_select,). Indices of the frames in X that are closest to the center of each candidate.
    agent_stakes (np.ndarray): array of shape (n_agents, n_select). Entry agent_stakes[i, j] indicates the stakes that agent i has on candidate j.
    '''
    
    # Select n_select candidates via least counts
    counts = Counter(kmeans.labels_)
    least_counts = np.asarray(counts.most_common()[::-1][:n_select])[:,0] # Which clusters contain lowest populations
    
    # Find frames closest to cluster centers of candidates
    least_counts_centers = kmeans.cluster_centers_[least_counts]
    central_frames_indices, _ = pairwise_distances_argmin_min(least_counts_centers, X)
    central_frames = X[central_frames_indices]
    
    # Compute agent stakes
    agent_stakes = np.zeros((n_agents, n_select))
    for candidate_idx, candidate in enumerate(least_counts):
        agent_indices = agent_idx[np.where(kmeans.labels_ == candidate)]
        agent_stakes[:, candidate_idx] = compute_agent_stakes(agent_indices, n_agents,  method=stakes_method, k=stakes_k)
        
    return central_frames, central_frames_indices, agent_stakes

def compute_agent_stakes(agent_indices, n_agents, method='percentage', k=None):
    '''
    Returns agent stakes of a cluster given the number of frames from each agent that fall in said cluster.
    
    Args
    -------------
    agent_indices (np.ndarray): array containing the agent index for each frame in the given cluster.
    n_agents (int): number of agents.
    method (str): one of "percentage", "max", "equal", or "logistic". 
        percentage: stakes are proportional to number of frames that agents have in the cluster.
        max: the agent with the max number of frames in the cluster has all the stakes.
        equal: all agents with at least one frame in the cluster have the same stakes.
        logistic: percentage stakes are transformed as 1/(1+e^(-k(x-0.5))) and renormalized. Must set parameter k (higher k ~= max, lower k ~= equal).
    k (float): parameter for logistic method. Ignored if method is not set to logistic.
    
    Returns
    -------------
    agent_stakes (np.ndarray): array of shape (n_agents, n_select). Entry agent_stakes[i, j] indicates the stakes that agent i has on candidate j.
    '''
    if (method=="logistic"):
        try: 
            assert(k is not None)
        except AssertionError as err:
            raise AssertionError("k parameter must be set if using method='logistic'") from err
    
    agent_indices_count = Counter(agent_indices)
    stakes = np.zeros(n_agents)
    total = sum(agent_indices_count.values())
    for agent, count in agent_indices_count.most_common():
        stakes[agent] = count/total
    
    if (method=="percentage"):
        pass
        
    elif (method=="logistic"):
        x0 = 0.5
        logistic_fun = lambda x: 1/(1 + np.exp(-k*(x-x0)))

        stakes_transformed = logistic_fun(stakes)
        stakes_transformed[np.where(stakes < 1e-18)] = 0 # Make sure that the function evaluates to 0 at x=0
        stakes_transformed /= stakes_transformed.sum() # Re-normalize
        stakes = stakes_transformed
    
    elif (method=="equal"):
        stakes[np.where(stakes!=0)] = 1/np.count_nonzero(stakes) 
    
    elif (method=="max"):
        # If two or more agents have the same number of frames (and these are equal to the maximum), one of them is picked arbitrarily to get stakes 1
        stakes[np.argmax(stakes)] = 1
        stakes[np.where(stakes < 1)] = 0
        
    else:
        raise ValueError("Method "+method+" not understood.") 
    
    try: 
        assert(np.abs(stakes.sum()-1) < 1e-18)
    except AssertionError as err:
        print("Stakes:", stakes)
        print("Method:", method)
        print("Agent indices:", agent_indices)
        raise AssertionError() from err
        
    return stakes

def compute_structure_reward_MA_standard_Euclidean(X_a, central_frames, stakes_a, weights_a, n_select):
    '''
    Computes the reward for each structure w.r.t. the given agent and returns it as an array of shape (n_select,).
    
    Args
    -------------
    X_a (np.ndarray): array of shape (?, n_features). Frames observed by a given agent.
    central_frames (np.ndarray): array of shape (n_select, n_features). Frames in X that are closest to the center of each candidate.
    stakes_a (np.ndarray): array of shape (n_select,). Stakes of the agent in each candidate.
    weights_a (np.ndarray): array of shape (n_features,). Weights that the given agent assigns to each order parameter.
    n_select (int): how many candidates were selected based on least counts policy.
    
    Returns
    -------------
    rewards (np.ndarray): array of shape (n_select,). Rewards assigned to candidates by given agent.
    '''
    # Initialize array
    rewards = np.empty(n_select)

    # Compute distribution parameters for frames observed by the agent
    mu = X_a.mean(axis=0)[:2]
    sigma = X_a.std(axis=0)[:2]
    
    # Compute distance of each candidate to the mean
    distances = (weights_a*np.abs(central_frames[:,:2] - mu)/sigma).sum(axis=1) # Shape is (n_select,)
    
    # Compute rewards
    # Since stakes_a is zero for those clusters that do not involve agent a, rewards are accurately set to zero here.
    rewards = stakes_a*distances
    
    return rewards

def compute_cumulative_reward_MA_standard_Euclidean(X, agent_idx, agent_stakes, central_frames_indices, n_select, n_agents, weights, which_agent):
    '''
    Returns the cumulative reward for current weights and a callable to the cumulative reward function (necessary to finetune weights).
    Note that this is the cumulative reward and reward function for the given agent.
    In the multi-agent implementation, this function is called `n_agents` times during the optimization step.
    
    Args
    -------------
    X (np.ndarray): array of shape (n_frames, n_features). Representative subset of frames from trajectories.
    agent_idx (np.ndarray): array of shape (n_frames,) indicating which agent originated each frame.
    agent_stakes (np.ndarray): array of shape (n_agents, n_select). Entry agent_stakes[i, j] indicates the stakes of agent i on cluster j.
    central_frames_indices (np.ndarray): array of shape (n_select,). Indices of the frames in X that are closest to the center of each candidate.
    n_select (int): how many candidates were selected based on least counts policy.
    n_agents (int): number of agents.
    weights (np.ndarray): array of shape (n_agents, n_features). Weights that each agent assigns to each order parameter.
    which_agent (int): number in [0, n_agents). Indicates which agent is computing the rewards.
    
    Returns
    -------------
    R (float): cumulative reward for selected agent.
    rewards_function (callable): reward function to maximize.  
    '''
    assert(which_agent < n_agents)
    
    # Access relevant frames
    central_frames = X[central_frames_indices]
    
    # Acess data for specific agent
    a = which_agent
    weights_a = weights[a]
    stakes_a = agent_stakes[a]
    
    indices = np.where(agent_idx == a)
    X_a = X[indices]
    
    def rewards_function(w):
        r = compute_structure_reward_MA_standard_Euclidean(X_a, central_frames, stakes_a, w, n_select)
        R = r.sum()
        return R
    
    R = rewards_function(weights_a)
    
    return R, rewards_function

def tune_weights_MA_standard_Euclidean(rewards_function, weights_a, delta=0.02):
    '''
    Defines constraints for optimization and maximizes rewards function. Returns OptimizeResult object.
    This function is called once per each agent per epoch.
    
    Args
    -------------
    rewards_function (callable): reward function to maximize. This corresponds to the reward function for a given agent.
    weights_a (np.ndarray): array of shape (n_features,). Weights that the given agent assigns to each order parameter.
    delta (float), default 0.02: maximum amount by which an entry in the weights matrix can change. Think of it as an upper-bound for the learning rate.
    
    Returns
    -------------
    results (scipy.optimize.OptimizeResult): result of the optimization for the weights of the given agent.
    '''
    weights_prev = weights_a
    
    # Create constraints
    constraints = []
    
    # Inequality constraints (fun(x, *args) >= 0)
    # This constraint makes the weights change by delta (at most)
    constraints.append({
        'type': 'ineq', 
        'fun': lambda weights, weights_prev, delta: delta - np.abs((weights_prev - weights)), 
        'jac': lambda weights, weights_prev, delta: np.diagflat(np.sign(weights_prev - weights)),
        'args': (weights_prev, delta),
    })
    
    # This constraint makes the weights be always positive
    constraints.append({
        'type': 'ineq', 
        'fun': lambda weights: weights, 
        'jac': lambda weights: np.eye(weights.shape[0]),
    })
    
    # Equality constraints (fun(x, *args) = 0)
    # This constraint makes sure the weights add up to one
    constraints.append({
        'type': 'eq',
        'fun': lambda weights: weights.sum()-1,
        'jac': lambda weights: np.ones(weights.shape[0]),
    })
    
    results = minimize(lambda x: -rewards_function(x), weights_prev,  method='SLSQP', constraints=constraints)
    
    return results

def select_starting_points_MA_standard_Euclidean(X, central_frames, agent_idx, agent_stakes, weights, n_agents, n_chosen=10, collaborative=True, competitive=False):
    '''
    Select starting positions for new simulations. Here, the collective rewards are computed as the sum of the rewards from each agent.
    
    Args
    -------------
    X (np.ndarray): array of shape (n_frames, n_features). Representative subset of frames from trajectories.
    central_frames (np.ndarray): array of shape (n_select, n_features). Frames that are closest to the cluster center of each candidate.
    agent_idx (np.ndarray): array of shape (n_frames,) indicating which agent originated each frame.
    agent_stakes (np.ndarray): array of shape (n_agents, n_select). Entry agent_stakes[i, j] indicates the fraction of frames from 
        candidate j that were generated by agent i.
    weights (list[np.ndarray]): list of length n_agents of shape (n_features, n_features). Weights that each agent assigns to each entry in the
        order parameter covariance matrix.
    n_agents (int): number of agents.
    n_chosen (int): number of candidates that will be used for new trajectories.
    collaborative (Bool): if set to True (default), the rewards from all agents are added together. 
        If False, only the reward from the agent who returned the highest value is used.
    competitive (Bool): this option is only taken into account if collaborative is set to False. 
        If True, the reward is the difference between the non-collaborative reward minus the rewards assigned by all other agents.
    
    Returns
    -------------
    chosen_frames (np.ndarray): array of shape (n_chosen, n_features). Frames from which to launch new trajectories.
    executors (np.ndarray): array of shape (n_chosen,). Index of the agent that launches each trajectory.
    '''
    
    # Compute collective rewards for each candidate with the updated weights
    n_select = len(central_frames)
    
    if collaborative:
        rewards = np.zeros(n_select)
        for a in range(n_agents):
            # Acess data for specific agent
            weights_a = weights[a]
            stakes_a = agent_stakes[a]

            indices = np.where(agent_idx == a)
            X_a = X[indices]
            rewards += compute_structure_reward_MA_standard_Euclidean(X_a, central_frames, stakes_a, weights_a, n_select)
    else:
        rewards_agent = np.zeros((n_agents, n_select))
        for a in range(n_agents):
            # Acess data for specific agent
            weights_a = weights[a]
            stakes_a = agent_stakes[a]

            indices = np.where(agent_idx == a)
            X_a = X[indices]
            rewards_agent[a] = compute_structure_reward_MA_standard_Euclidean(X_a, central_frames, stakes_a, weights_a, n_select)
        rewards_max = np.max(rewards_agent, axis=0)

        if competitive:
            rewards = 2*rewards_max - np.sum(rewards_agent, axis=0)
        else:
            rewards = rewards_max
    
    assert(len(rewards) == n_select)
    indices = np.argsort(rewards)[-n_chosen:][::-1] # Indices of frames with maximum reward
    chosen_frames = central_frames[indices] # Frames that will be used to start new simulations
    executors = agent_stakes.argmax(axis=0)[indices] # Agents that will run the new trajectories 
    
    return chosen_frames, executors

def spawn_trajectories_MA(trajectories, chosen_frames, executors, seed=None, traj_len=500, potential=''):
    '''
    Spawns new trajectories and assigns them to the correct agent.
    
    Args
    -------------
    trajectories (list[list[np.ndarray]]): trajectories collected so far. They should be accessed as trajectories[ith_agent][jth_trajectory].
    chosen_frames (np.ndarray): array of shape (n_chosen, n_features). Frames from which to launch new trajectories.
    executors (np.ndarray): array of shape (n_chosen,). Index of the agent that launches each trajectory.
    
    Returns
    -------------
    trajectories (list[list[np.ndarray]]): trajectories collected so far (inlcuding new ones). 
        They should be accessed as trajectories[ith_agent][jth_trajectory].
    '''
    
    for frame, agent in zip(chosen_frames, executors):
        new_traj = run_trajectory(n_steps=traj_len, potential=potential, initial_position=frame)
        trajectories[agent].append(new_traj)
        
    return trajectories

def collect_initial_data(num_trajectories, traj_len, potential, initial_positions, trajectories):
   '''
   Collect some initial data before using adaptive sampling.
   '''
   for _ in range(num_trajectories):
       for i, initial_position in enumerate(initial_positions):
           traj = run_trajectory(n_steps=traj_len, potential=potential, initial_position=initial_position)
           trajectories[i].append(traj)
   
   return trajectories

def run_trial(potential, initial_positions, epochs, output_dir='', output_prefix='', **kwargs):
    '''
    Runs a trial of MA REAP with standard Euclidean distance rewards.
    
     Args
    -------------
    potential (str): potential on which to run the trial (currently only two-dimensional potentials are used).
    initial_positions (list[np.ndarray]): starting points for simulations. Lenght of list must match number of agents.
    epochs (int): specifies for how many epochs to run a trial.
    output_dir (str): folder where to store the results (it will be created in current working directory if it does not exist).
    output_prefix (str): common prefix for all log files.
    **kwargs: used to specify model hyperparameters. Must include following keys:
        num_spawn (int): number of total trajectories to spawn per epoch.
        n_select (int): number of least-count candidates selected per epoch.
        n_agents (int): number of agents.
        traj_len (int): length of each trajectory ran.
        delta (float): upper boundary for learning step.
        n_features (int): number of collective variables. (Currently ignored because only two-dimensional systems are used.)
        d (int): dimensionality of the slow manifold (used to compute number of clusters to use). (Should be two for these trials.)
        gamma (float): parameter to determine number of clusters (theoretically in [0.5, 1]).
        b (float): parameter to determine number of clusters.
        max_frames (int): maximum number of frames to use in clustering steps (take random subsample to accelerate testing).
        xlim (tuple of (x_min, x_max, x_stride)): used to define grid where are exlpored will be computed. (Will be replaced in the future to accommodate arbitrary dimensions.)
        ylim (tuple of (y_min, y_max, y_stride)): used to define grid where are exlpored will be computed. (Will be replaced in the future to accommodate arbitrary dimensions.)
        threshold (float): any area above this free energy threshold will be ignored in the computation of explored area.
        potential_func (callable): necessary to compute explored area.
        stakes_method (str): one of "percentage", "max", "equal", or "logistic". 
            percentage: stakes are proportional to number of frames that agents have in the cluster.
            max: the agent with the max number of frames in the cluster has all the stakes.
            equal: all agents with at least one frame in the cluster have the same stakes.
            logistic: percentage stakes are transformed as 1/(1+e^(-k(x-0.5))) and renormalized. Must set parameter k (k >= 100 is similar to max, k <= 0.01 is similar to equal).
        stakes_k (float): parameter for logistic method for stakes calculation. Ignored if method is not set to logistic.
        collaborative (Bool): whether to use collaborative or non-collaborative reward combination (incompatible with competitive=True).
        competitive (Bool): whether to use competitive reward combination (incompatible with collaborative=True).
    
    Returns
    -------------
    None. Results are saved to output_dir.
    '''
    # Step 1: define some hyperparameters and initialize arrays --> To be provided via init_variables
    num_spawn = kwargs['num_spawn'] # Number of trajectories spawn per epoch
    n_select = kwargs['n_select'] # Number of least-count candidates selected per epoch
    n_agents = kwargs['n_agents']
    traj_len = kwargs['traj_len']
    delta = kwargs['delta'] # Upper boundary for learning step
    n_features = kwargs['n_features'] # Number of variables in OP space (in this case it's x, y, z where z is irrelevant to the potential)
    d = kwargs['d'] # Parameter to determine number of clusters (intrinsic dimensionality given by potential function)
    gamma = kwargs['gamma'] # Parameter to determine number of clusters
    b = kwargs['b'] # Parameter to determine number of clusters
    max_frames = kwargs['max_frames']
    xlim = kwargs['xlim']
    ylim = kwargs['ylim']
    threshold = kwargs['threshold']
    debug = kwargs['debug']
    potential_func = kwargs['potential_func']
    stakes_method = kwargs['stakes_method']
    stakes_k = kwargs['stakes_k']
    collaborative = kwargs['collaborative']
    competitive = kwargs['competitive']
    
    # Step 2: set initial weights
    weights = [np.ones((n_features))/n_features for _ in range(n_agents)]
    
    # Step 3: collect some initial data
    trajectories = [[] for _ in range(n_agents)]
    trajectories = collect_initial_data(num_spawn, traj_len, potential, initial_positions, trajectories)
    
    # Steps 4-9: cluster, compute rewards, tune weights, run new simulations, and repeat

    # Logs
    least_counts_points_log = [] # For central frames from candidates
    agent_stakes_log = []
    cumulative_reward_log = [[] for _ in range(n_agents)]
    weights_log = [[] for _ in range(n_agents)]
    individual_rewards_log = [[] for _ in range(n_agents)]
    selected_structures_log = []
    area_log = []
    area_agents_log = [[] for _ in range(n_agents)]
    executors_log = []
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for e in range(epochs):
        print("Running epoch: {}/{}".format(e+1, epochs), end='\r', flush=True)

        # Clustering
        kmeans, X, agent_idx = clustering_MA(trajectories, n_agents, n_select, max_frames=max_frames, b=b, gamma=gamma, d=d)

        # Select candidates
        central_frames, central_frames_indices, agent_stakes = select_least_counts_MA(kmeans, X, agent_idx, n_agents, stakes_method=stakes_method, stakes_k=stakes_k, n_select=n_select)

        # Save logs
        least_counts_points_log.append(central_frames)
        agent_stakes_log.append(agent_stakes)

        # Compute rewards and tune weights (this step is done agent-wise)
        for a in range(n_agents):
            # Compute reward and optimize weights
            R, reward_fun = compute_cumulative_reward_MA_standard_Euclidean(X, agent_idx, agent_stakes, central_frames_indices, n_select, n_agents, weights, a)
            optimization_results = tune_weights_MA_standard_Euclidean(reward_fun, weights[a], delta=delta)

            # Check if optimization worked
            if optimization_results.success:
                pass
            else:
                print("ERROR: CHECK OPTIMIZATION RESULTS FOR AGENT", a)
                print(optimization_results)
                break

            # Set new weights
            weights[a] = optimization_results.x

            # Compute individual rewards for storage
            indices = np.where(agent_idx == a)
            X_a = X[indices]
            individual_rewards = compute_structure_reward_MA_standard_Euclidean(X_a, central_frames, agent_stakes[a], weights[a], n_select) # Rewards for each candidate

            # Update logs
            cumulative_reward_log[a].append(R)
            weights_log[a].append(weights[a])
            individual_rewards_log[a].append(individual_rewards)

        chosen_frames, executors = select_starting_points_MA_standard_Euclidean(X, central_frames, agent_idx, agent_stakes, weights, n_agents, n_chosen=num_spawn, collaborative=collaborative, competitive=competitive)

        selected_structures_log.append(chosen_frames)
        executors_log.append(executors)

        trajectories = spawn_trajectories_MA(trajectories, chosen_frames, executors, potential=potential)
        
        ### Compute area explored ###
        concatenated_trajs = np.concatenate([trajectories[i][j] for i in range(n_agents) for j in range(len(trajectories[i]))])[:, :2]
        area_log.append(area_explored(potential_func, xlim, ylim, concatenated_trajs, threshold))
        for a in range(n_agents):
            concatenated_trajs = np.concatenate([trajectories[a][j] for j in range(len(trajectories[a]))])[:, :2]
            area_agents_log[a].append(area_explored(potential_func, xlim, ylim, concatenated_trajs, threshold))

        
        if debug: # Plotting of cross potential (must readjust parameters for other landscapes)
            if (e % 25 == 0) or (e+1 == epochs):
                print('Weights:', weights, flush=True)
                print('Area explored:', area_log[-1], flush=True)
                x_plot = np.arange(*xlim)
                y_plot = np.arange(*ylim)
                X_plot, Y_plot = np.meshgrid(x_plot, y_plot) # grid of point
                Z_plot = potential_func(X_plot, Y_plot) # evaluation of the function on the grid
                im = plt.imshow(Z_plot, cmap=plt.cm.jet, extent=[xlim[0], xlim[1], ylim[0], ylim[1]]) # drawing the function
                plt.colorbar(im) # adding the colobar on the right
                for a in range(n_agents):
                    plt.scatter(np.concatenate(trajectories[a])[::3,0], np.concatenate(trajectories[a])[::3,1], s=0.5)
                plt.scatter(central_frames[:,0], central_frames[:,1], s=20, alpha=0.5, c='purple')
                plt.scatter(chosen_frames[:,0], chosen_frames[:,1], s=1, alpha=1, c='black')
                plt.xlim([0, 2])
                plt.ylim([0, 2])
                plt.savefig(os.path.join(output_dir, output_prefix + 'landscape_epoch_{}.png'.format(e+1)), dpi=150)
                plt.close()
    
    ### Save results ###
    save_pickle(trajectories, os.path.join(output_dir, output_prefix + 'trajectories.pickle'))
    save_pickle(weights_log, os.path.join(output_dir, output_prefix + 'weights_log.pickle'))
    save_pickle(least_counts_points_log, os.path.join(output_dir, output_prefix + 'least_counts_points_log.pickle'))
    save_pickle(cumulative_reward_log, os.path.join(output_dir, output_prefix + 'cumulative_reward_log.pickle'))
    save_pickle(individual_rewards_log, os.path.join(output_dir, output_prefix + 'individual_rewards_log.pickle'))
    save_pickle(selected_structures_log, os.path.join(output_dir, output_prefix + 'selected_structures_log.pickle'))
    save_pickle(agent_stakes_log, os.path.join(output_dir, output_prefix + 'agent_stakes_log.pickle'))
    save_pickle(area_agents_log, os.path.join(output_dir, output_prefix + 'area_agents_log.pickle'))
    save_pickle(area_log, os.path.join(output_dir, output_prefix + 'area_log.pickle'))
    save_pickle(executors_log, os.path.join(output_dir, output_prefix + 'executors_log.pickle'))