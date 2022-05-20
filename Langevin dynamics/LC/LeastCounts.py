'''
This document implements the least counts policy using kmeans clustering (no reward or directional component).
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

def clustering(trajectories, n_select, n_clusters=None, max_frames=1e5, b=1e-4, gamma=0.7, d=None):
    '''
    Clusters all (or a representative subset) of the frames in trajectories using KMeans. Returns clustering object, which will be used to select the 
    least counts clusters. The agents that created each frame are remembered and their indices are returned as well. The selected subset of the total
    frames are also returned.
    
    Args
    -------------
    trajectories (list[np.ndarray]): trajectories collected so far. They should be accessed as trajectories[ith_trajectory].
    n_clusters (int), default None: number of clusters to use for KMeans. If None, a heuristic by Buenfil et al. (2021) is used to approximate the number of clusters needed.
    max_frames (int), default 1e5: maximum number of frames to use in the clustering step. If set to 0 or None, all frames are used.
    b (float), default 1e-4: coefficient for n_clusters heuristic.
    gamma (float), default 0.7: exponent for n_clusters heuristic (should theoretically be in [0.5, 1]).
    d (int), no default: intrinsic dimensionality of slow manifold for the system. Used in n_clusters heuristic. Must be defined.
   
    Returns
    -------------
    KMeans (sklearn.cluster.KMeans): fitted KMeans clustering object.
    X (np.ndarray): array of shape (max_frames, n_features) containing the subset of the data used for clustering.
    '''
    
    # Put frames in format that is usable for KMeans
          
    trajectory = np.concatenate(trajectories)
    total_frames = len(trajectory)
    
    # Downsample number of points
    if (not max_frames) or (total_frames <= max_frames):
        X = trajectory        
    
    elif total_frames > max_frames:
        max_frames = int(max_frames)
        rng = np.random.default_rng()
        rand_indices = rng.choice(len(trajectory), max_frames, replace=False)
        X = trajectory[rand_indices]
        
    # Use heuristic from https://openreview.net/pdf?id=00thAjcutwh to determine number of clusters
    if (n_clusters is None):
        n_clusters = int(b*(min(total_frames, max_frames)**(gamma*d)))
    
    if (n_clusters < n_select):
        n_clusters = n_select # Make sure there are enough clusters to select candidates from
        
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, init='random').fit(X)
    
    return kmeans, X

def select_least_counts(kmeans, X, n_select=50):
    '''
    Select candidate clusters for new round of simulations based on least counts policy.
    
    Args
    -------------
    kmeans (sklearn.cluster.KMeans): KMeans clustering object fitted on X.
    X (np.ndarray): array of shape (n_frames, n_features). Representative subset of frames from trajectories.
    n_select (int), default 50: how many candidates to select based on least counts policy.
    
    Returns
    -------------
    central_frames (np.ndarray): array of shape (n_select, n_features). Frames in X that are closest to the center of each candidate.
    central_frames_indices (np.ndarray): array of shape (n_select,). Indices of the frames in X that are closest to the center of each candidate.
    '''
    
    # Select n_select candidates via least counts
    counts = Counter(kmeans.labels_)
    least_counts = np.asarray(counts.most_common()[::-1][:n_select])[:,0] # Which clusters contain lowest populations
    
    # Find frames closest to cluster centers of candidates
    least_counts_centers = kmeans.cluster_centers_[least_counts]
    central_frames_indices, _ = pairwise_distances_argmin_min(least_counts_centers, X)
    central_frames = X[central_frames_indices]
        
    return central_frames, central_frames_indices

def spawn_trajectories(trajectories, chosen_frames, seed=None, traj_len=500, potential=''):
    '''
    Spawns new trajectories.
    
    Args
    -------------
    trajectories (list[np.ndarray]): trajectories collected so far. They should be accessed as trajectories[ith_trajectory].
    chosen_frames (np.ndarray): array of shape (n_chosen, n_features). Frames from which to launch new trajectories.
    
    Returns
    -------------
    ttrajectories (list[np.ndarray]): trajectories collected so far (including new ones). They should be accessed as trajectories[ith_trajectory].
    '''
    
    for frame in chosen_frames:
        new_traj = run_trajectory(n_steps=traj_len, potential=potential, initial_position=frame)
        trajectories.append(new_traj)
        
    return trajectories

def collect_initial_data(num_trajectories, traj_len, potential, initial_positions, trajectories):
    '''
    Collect some initial data before using adaptive sampling.
    '''
    for _ in range(num_trajectories):
        for i, initial_position in enumerate(initial_positions):
            traj = run_trajectory(n_steps=traj_len, potential=potential, initial_position=initial_position)
            trajectories.append(traj)
    
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
        num_spawn (int): number of total trajectories to spawn per epoch. (For these trials, it is equal to n_select because we launch one trajectory from selected cluster.)
        n_select (int): number of least-count clusters selected per epoch.
        traj_len (int): length of each trajectory ran.
        n_features (int): number of collective variables. (Currently ignored because only two-dimensional systems are used.)
        d (int): dimensionality of the slow manifold (used to compute number of clusters to use). (Should be two for these trials.)
        gamma (float): parameter to determine number of clusters (theoretically in [0.5, 1]).
        b (float): parameter to determine number of clusters.
        max_frames (int): maximum number of frames to use in clustering steps (take random subsample to accelerate testing).
        xlim (tuple of (x_min, x_max, x_stride)): used to define grid where area exlpored will be computed. (Will be replaced in the future to accommodate arbitrary dimensions.)
        ylim (tuple of (y_min, y_max, y_stride)): used to define grid where area exlpored will be computed. (Will be replaced in the future to accommodate arbitrary dimensions.)
        threshold (float): any area above this free energy threshold will be ignored in the computation of explored area.
        potential_func (callable): necessary to compute explored area.
    
    Returns
    -------------
    None. Results are saved to output_dir.
    '''
    # Step 1: define some hyperparameters and initialize arrays --> To be provided via init_variables
    num_spawn = kwargs['n_select'] # Number of trajectories spawn per epoch --> Set equal to n_select for these trials
    n_select = kwargs['n_select'] # Number of least-count candidates selected per epoch
    traj_len = kwargs['traj_len']
    n_features = kwargs['n_features'] # Number of variables in OP space
    d = kwargs['d'] # Parameter to determine number of clusters (intrinsic dimensionality given by potential function)
    gamma = kwargs['gamma'] # Parameter to determine number of clusters
    b = kwargs['b'] # Parameter to determine number of clusters
    max_frames = kwargs['max_frames']
    xlim = kwargs['xlim']
    ylim = kwargs['ylim']
    threshold = kwargs['threshold']
    debug = kwargs['debug']
    potential_func = kwargs['potential_func']
    
    # Step 2: collect some initial data
    trajectories = []
    trajectories = collect_initial_data(num_spawn, traj_len, potential, initial_positions, trajectories)
     
    # Step 3: cluster, select starting points, run new simulations, and repeat

    # Logs
    least_counts_points_log = [] # For central frames from candidates
    area_log = []
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for e in range(epochs):
        print("Running epoch: {}/{}".format(e+1, epochs), end='\r', flush=True)

        # Clustering
        kmeans, X = clustering(trajectories, n_select, n_clusters=None, max_frames=max_frames, b=b, gamma=gamma, d=d)

        # Select candidates
        central_frames, central_frames_indices = select_least_counts(kmeans, X, n_select=n_select)

        # Save logs
        least_counts_points_log.append(central_frames)

        trajectories = spawn_trajectories(trajectories, central_frames, potential=potential)
        
        ### Compute area explored ###
        concatenated_trajs = np.concatenate(trajectories)[:, :2]
        area_log.append(area_explored(potential_func, xlim, ylim, concatenated_trajs, threshold))
        
        if debug: # Only works for cross potential
            if (e % 25 == 0) or (e+1 == epochs):
                print('Area explored:', area_log[-1], flush=True)
                x_plot = np.arange(*(-0.5, 2.5, 0.005))
                y_plot = np.arange(*(-0.5, 2.5, 0.005))
                X_plot, Y_plot = np.meshgrid(x_plot, y_plot) # grid of point
                Z_plot = potential_func(X_plot, Y_plot) # evaluation of the function on the grid
                im = plt.imshow(Z_plot, cmap=plt.cm.jet, extent=[xlim[0], xlim[1], ylim[0], ylim[1]]) # drawing the function
                plt.scatter(np.concatenate(trajectories)[::3,0], np.concatenate(trajectories)[::3,1], s=0.5)
                plt.colorbar(im) # adding the colobar on the right
                plt.scatter(central_frames[:,0], central_frames[:,1], s=20, alpha=0.2, c='purple')
                plt.xlim([0, 2])
                plt.ylim([0, 2])
                plt.savefig(os.path.join(output_dir, output_prefix + 'landscape_epoch_{}.png'.format(e+1)), dpi=150)
                plt.close()
    
    ### Save results ###
    save_pickle(trajectories, os.path.join(output_dir, output_prefix + 'trajectories.pickle'))
    save_pickle(least_counts_points_log, os.path.join(output_dir, output_prefix + 'least_counts_points_log.pickle'))
    save_pickle(area_log, os.path.join(output_dir, output_prefix + 'area_log.pickle'))
