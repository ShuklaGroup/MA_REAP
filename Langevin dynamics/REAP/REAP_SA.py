'''
This document implements the vanilla REAP algorithm using kmeans clustering and the standardized Euclidean distance in the reward definition.
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

def clustering(trajectories, n_select, n_clusters=None, max_frames=1e5, b=1, gamma=1, d=1):
    '''
    Clusters all frames in trajectories using KMeans.
    Returns clustering object.
    '''
    trajectories = np.asarray(trajectories)
    total_frames = trajectories.shape[0]*trajectories.shape[1]
    
    # Use heuristic from https://openreview.net/pdf?id=00thAjcutwh to determine number of clusters
    
    if (n_clusters is None):
        n_clusters = int(b*(min(total_frames, max_frames)**(gamma*d)))
    if (n_clusters < n_select):
        n_clusters = n_select
 
    X = trajectories.reshape((-1, 3)) # Turn trajectories into array of shape (n_samples, n_features)
    kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=5).fit(X)
    
    return kmeans

def select_least_counts(kmeans, trajectories, n_select):
    '''
    Select new starting clusters.
    Returns indices of n_select clusters with least counts and closest points to said clusters.
    '''
    counts = Counter(kmeans.labels_)
    least_counts = np.asarray(counts.most_common()[::-1][:n_select])[:,0] # Which clusters contain lowest populations
    
    trajectories = np.asarray(trajectories)
    X = trajectories.reshape((-1, 3)) # Turn trajectories into array of shape (n_samples, n_features)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    
    return least_counts, X[closest[least_counts]] 

def compute_structure_reward(trajectories, selected_points, weights):
    '''
    Computes the reward for each structure and returns it as an array.
    '''
    trajectories = np.asarray(trajectories)
    X = trajectories.reshape((-1, 3))
    mu = X.mean(axis=0)[:2] # Drop third dimension
    sigma = X.std(axis=0)[:2] # Drop third dimension
    selected_points = selected_points[:,:2] # Drop third dimension
    
    return (weights*np.abs(selected_points - mu)/sigma).sum(axis=1) # Shape is (selected_points.shape[0],)

def compute_cumulative_reward(trajectories, selected_points, weights):
    '''
    Returns the cumulative reward for current weights and a callable to the cumulative reward function (necessary to finetune weights).
    '''
    def rewards_function(w):
        r = compute_structure_reward(trajectories, selected_points, w)
        R = r.sum()
        return R
    
    R = rewards_function(weights)
    
    return R, rewards_function

def tune_weights(rewards_function, weights, delta=0.02):
    '''
    Defines constraints for optimization and maximizes rewards function. Returns OptimizeResult object.
    '''
    
    weights_prev = weights
    
    # Create constraints
    constraints = []
    
    # Inequality constraints (fun(x, *args) >= 0)
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
    constraints.append({
        'type': 'eq',
        'fun': lambda weights: weights.sum()-1,
        'jac': lambda weights: np.ones(weights.shape[0]),
    })
    
    results = minimize(lambda x: -rewards_function(x), weights_prev,  method='SLSQP', constraints=constraints)
    
    return results

def select_starting_points(rewards, selected_points, n_select=10):
    '''
    Select starting positions for new simulations.
    '''
    assert(rewards.shape[0] == selected_points.shape[0])
    indices = np.argsort(rewards)[-n_select:][::-1]
    
    return selected_points[indices]

def spawn_trajectories(N, K, trajectories, selected_actions, seed=None, traj_len=500, potential=''):
    '''
    
    Args:
        N (int): number of trajectories to spawn (not used yet).
        K (int): number of agents (not implemented yet).
        trajectories (np.ndarray): current trajectory data collected.
        selected_actions (np.ndarray): points from where new simulations will be restarted.
        seed (int): random seed.
        traj_len (int): trajectory steps.
        potential (str): potential for Langevin dynamics.
    '''
    starting_points = selected_actions # New simulations are started from the single structure that was selected for each cluster
    new_trajs = []
    for init_pos in starting_points:
        new_traj = run_trajectory(n_steps=traj_len, potential=potential, initial_position=init_pos)
        new_trajs.append(new_traj)
    trajectories.extend(new_trajs)
    
    return trajectories

def collect_initial_data(num_trajectories, traj_len, potential, initial_positions, trajectories):
   '''
   Collect some initial data before using adaptive sampling.
   '''
   for _ in range(num_trajectories):
       for i, initial_position in enumerate(initial_positions):
            traj = run_trajectory(n_steps=traj_len, potential=potential, initial_position=initial_position)
            trajectories[0].append(traj) # Forced to first and only agent
   
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
        n_agents (int): number of agents. --> In this implementation this is forced to be 1
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
    
    Returns
    -------------
    None. Results are saved to output_dir.
    '''
    # Step 1: define some hyperparameters and initialize arrays --> To be provided via init_variables
    num_spawn = kwargs['num_spawn'] # Number of trajectories spawn per epoch
    n_select = kwargs['n_select'] # Number of least-count candidates selected per epoch
    n_agents = 1 # Forced
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
    
    # Step 2: set initial weights
    weights = [np.ones((n_features))/n_features for _ in range(n_agents)]
    
    # Step 3: collect some initial data
    trajectories = [[] for _ in range(n_agents)]
    trajectories = collect_initial_data(num_spawn*2, traj_len, potential, initial_positions, trajectories) # Multiply by 2 to use same initial data as in MA REAP

    # Steps 4-9: cluster, compute rewards, tune weights, run new simulations, and repeat
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # Logs
    least_counts_points_log = [[] for _ in range(n_agents)]
    cumulative_reward_log = [[] for _ in range(n_agents)]
    weights_log = [[] for _ in range(n_agents)]
    individual_rewards_log = [[] for _ in range(n_agents)]
    selected_structures_log = [[] for _ in range(n_agents)]
    area_log = []
    area_agents_log = [[] for _ in range(n_agents)]

    for e in range(epochs):
        print("Running epoch: {}/{}".format(e+1, epochs), end='\r', flush=True)
        
        for i in range(n_agents):

            clusters = clustering(trajectories[i], n_select, max_frames=max_frames, b=b, gamma=gamma, d=d)
            
            least_counts_clusters, least_counts_points = select_least_counts(clusters, trajectories[i], n_select)

            least_counts_points_log[i].append(least_counts_points)

            R, reward_fun = compute_cumulative_reward(trajectories[i], least_counts_points, weights[i])

            cumulative_reward_log[i].append(R)

            optimization_results = tune_weights(reward_fun, weights[i], delta=delta)
            # Check if optimization worked
            if optimization_results.success:
                pass
            else:
                print("ERROR: CHECK OPTIMIZATION RESULTS")
                break

            weights[i] = optimization_results.x

            weights_log[i].append(weights[i])

            rewards = compute_structure_reward(trajectories[i], least_counts_points, weights[i])

            individual_rewards_log[i].append(rewards)

            new_starting_points = select_starting_points(rewards, least_counts_points, n_select=num_spawn)

            selected_structures_log[i].append(new_starting_points)

            trajectories[i] = spawn_trajectories(0, 0, trajectories[i], new_starting_points, seed=None, traj_len=traj_len, potential=potential)

        ### Compute area explored ###
        concatenated_trajs = np.concatenate([trajectories[i][j] for i in range(n_agents) for j in range(len(trajectories[i]))])[:, :2]
        area_log.append(area_explored(potential_func, xlim, ylim, concatenated_trajs, threshold))

        if debug: # Only works for cross potential
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
                    plt.scatter(least_counts_points_log[a][-1][:,0], least_counts_points_log[a][-1][:,1], s=20, alpha=0.2, c='purple')
                    plt.scatter(selected_structures_log[a][-1][:,0], selected_structures_log[a][-1][:,1], s=1, alpha=1, c='black')
                plt.xlim([0, 2])
                plt.ylim([0, 2])
                plt.savefig(os.path.join(output_dir, output_prefix + 'landscape_epoch_{}.png'.format(e+1)), dpi=150)
                plt.close()
                
        for a in range(n_agents):
            concatenated_trajs = np.concatenate([trajectories[a][j] for j in range(len(trajectories[a]))])[:, :2]
            area_agents_log[a].append(area_explored(potential_func, xlim, ylim, concatenated_trajs, threshold))
    
    ### Save results ###
    save_pickle(trajectories, os.path.join(output_dir, output_prefix + 'trajectories.pickle'))
    save_pickle(weights_log, os.path.join(output_dir, output_prefix + 'weights_log.pickle'))
    save_pickle(least_counts_points_log, os.path.join(output_dir, output_prefix + 'least_counts_points_log.pickle'))
    save_pickle(cumulative_reward_log, os.path.join(output_dir, output_prefix + 'cumulative_reward_log.pickle'))
    save_pickle(individual_rewards_log, os.path.join(output_dir, output_prefix + 'individual_rewards_log.pickle'))
    save_pickle(selected_structures_log,os.path.join(output_dir, output_prefix + 'selected_structures_log.pickle'))
    save_pickle(area_agents_log, os.path.join(output_dir, output_prefix + 'area_agents_log.pickle'))
    save_pickle(area_log, os.path.join(output_dir, output_prefix + 'area_log.pickle'))
