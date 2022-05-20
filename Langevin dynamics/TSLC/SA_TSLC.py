import os
import numpy as np
import matplotlib.pyplot as plt
import openmm as mm
from utils import save_pickle, load_pickle, plot_potential, run_trajectory, setup_simulation, area_explored
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
from collections import Counter
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist, squareform

def define_colvars(trajectory):
    '''
    Takes in raw trajectory and returns a trajectory in collective variables space. In this example, it takes the 3d positions of the
    only particle and returns corresponding vectors (x, y, z, arctan(y/x)) with appropriate signs.
    
    Args
    -------------
    trajectory (np.ndarray): trajectory of shape (n_frames, 3).
    
    Returns
    -------------
    colvar_traj (np.ndarray): trajectory in collective variables space of shape (n_frames, 4).
    '''
    
    colvar = np.arctan2(trajectory[:,1], trajectory[:,0])
    colvar[colvar < 0] += 2*np.pi
    covlar = np.degrees(colvar)
    colvar_traj = np.append(trajectory, colvar.reshape(-1, 1), axis=1)
    
    return colvar_traj

def define_colvars_grad(point):
    '''
    Returns the gradient in the collective variables space w.r.t. particle position.
    
    Args
    -------------
    point (np.ndarray): position of shape (3,).
    
    Returns
    -------------
    grad (np.ndarray): gradient in collective variables space of shape (3, 4).
    '''
    x, y, z = point
    dx = np.asarray([1, 0, 0, -y/(x**2 + y**2)])
    dy = np.asarray([0, 1, 0, x/(x**2 + y**2)])
    dz = np.asarray([0, 0, 1, 0])
    
    grad = np.vstack([dx, dy, dz])
    
    return grad

def principal_components(points, d=None):
    '''
    Returns the principal components of the given points. 
    To be used on the points belonging to each of the least counts candidates clusters.
    
    Args
    -------------
    points (np.ndarray): points in real space belonging to the same cluster. Shape (n_points, 3).
    d (int): number of principal components to return. 
        This parameter should be the dimensionality of the slow manifold that characterizes the potential.
    
    Returns
    -------------
    pcs (np.ndarray): d principal components. Shape (d, 3).
    '''
    pca = PCA(n_components=d)
    pca_trans = pca.fit(points)
    
    return pca_trans.components_

def clustering_MA_FFT(trajectories, n_agents, n_select, n_clusters=None, stakes_method='percentage', stakes_k=None, max_frames=None, b=1e-4, gamma=0.7, d=None, l=2):
    '''
    Clusters all (or a representative subset) of the frames in trajectories using Fastest First Traversal. 
    Returns clustering object, which will be used to select the least counts clusters. The agents that 
    created each frame are remembered and their indices are returned as well. The selected subset of the total
    frames are also returned. Gis, Vis, and agent stakes on all clusters are also computed here (for TSLC).
    
    Args
    -------------
    trajectories (list[list[np.ndarray]]): trajectories collected so far. They should be accessed as trajectories[ith_agent][jth_trajectory].
        Trajectories are internally projected to collective variable space for clustering.
    n_agents (int): number of agents.
    n_clusters (int), default None: number of clusters to use for KMeans. If None, a heuristic by Buenfil et al. (2021) is used to approximate the number of clusters needed.
    max_frames (int), default 1e5: maximum number of frames to use in the clustering step. If set to 0 or None, all frames are used.
    b (float), default 1e-4: coefficient for n_clusters heuristic.
    gamma (float), default 0.7: exponent for n_clusters heuristic (should theoretically be in [0.5, 1]).
    d (int), no default: intrinsic dimensionality of slow manifold for the system. Used in n_clusters heuristic. Must be defined.
    l (int), default 2: constant for n_clusters heuristic.
   
    Returns
    -------------
    colvar_traj (list[list[np.ndarray]]): trajectories collected so far in collective variable space.
    KMeans (sklearn.cluster.KMeans): fitted KMeans clustering object. Note that the clustering occurs in collective variable space.
    X (np.ndarray): array of shape (max_frames, D) containing the subset of the data used for clustering (euclidean space).
    X_colvar (np.ndarray): array of shape (max_frames, n_features) containing the subset of the data used for clustering (colvar space).
    agent_idx (np.ndarray): array of shape (max_frames,) containing the index of the agent that originated each frame.
    Gis
    Vis (list[np.ndarrays]): list of Vi matrices. Order is identical to indexing of clusters.
    agent_stakes
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
    colvar_traj = define_colvars(trajectory)
    
    assert(len(agent_index) == len(trajectory))
    
    # Downsample number of points
    if (not max_frames) or (total_frames <= max_frames):
        X = trajectory
        X_colvar = colvar_traj
        agent_idx = agent_index
        max_frames = total_frames
    
    elif total_frames > max_frames:
        max_frames = int(max_frames)
        rng = np.random.default_rng()
        rand_indices = rng.choice(len(trajectory), max_frames, replace=False)
        X = trajectory[rand_indices]
        X_colvar = colvar_traj[rand_indices]
        agent_idx = agent_index[rand_indices]
       
    # Use heuristic from https://openreview.net/pdf?id=00thAjcutwh to determine number of clusters
    if (n_clusters is None):
        n_clusters = int(b*(max_frames**(gamma*d)))
    if (n_clusters < n_select):
        n_clusters = n_select
    
    # First clustering attempt using FFT
    centroids = _FFT_helper(X_colvar, n_clusters, l=l)
    kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=1, init=centroids).fit(X_colvar)
    
    # If first attempt fails, repeat until convergence is reached
    while len(np.unique(kmeans.labels_)) != n_clusters:
        centroids = _FFT_helper(X_colvar, n_clusters, l=l)
        kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=1, init=centroids).fit(X_colvar)
    
    # Used to compute Gis, Vis, and agent stakes
    # Gis
    central_frames_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_colvar)
    central_frames = X[central_frames_indices]
    Gis = compute_Gi_all_clusters(central_frames)
    
    # Vis
    Vis = compute_Vi_all_clusters(kmeans, X, d)
    
    # Agent stakes
    agent_stakes = compute_agent_stakes(kmeans, agent_idx, n_agents, method=stakes_method, k=stakes_k)
#     print(agent_stakes)
    return kmeans, X, X_colvar, agent_idx, Gis, Vis, agent_stakes

def _FFT_helper(X_colvar, n_clusters, l=2):
    '''
    Helper for clustering_MA_FFT function. Code based on implementation by James Buenfil et al.
    '''
    n_clusters_prime = int(n_clusters*np.log(n_clusters) + l)
    rng = np.random.default_rng()
    rand_indices = rng.choice(len(X_colvar), n_clusters_prime, replace=False)
    init_points = X_colvar[rand_indices]
 
    mu_1 = 0 # Choosing first one is the same as picking at random
    mus = [ mu_1 ]
    available_mus = [ i for i in range(1, n_clusters_prime) ]
   
    D = squareform(pdist(init_points)) # Euclidean norm distance matrix
    np.fill_diagonal(D, np.inf) 
 
    #### Vectorize highest minimum search ####
    for _ in range(1, n_clusters):
        #an index corresponding to max min distance point so far
        max_sofar = available_mus[0]
        min_dist = np.amin(D[max_sofar, np.asarray(mus)])

        for i in available_mus[1:]:
            if np.amin(D[i, np.array(mus)]) > min_dist:
                max_sofar = i
                min_dist = np.amin(D[max_sofar, np.array(mus)])
        available_mus.remove(max_sofar)
        mus.append(max_sofar)
    mus = np.asarray(mus)
    
    return X_colvar[rand_indices[mus]]

def select_least_counts_MA(kmeans, X, X_colvar, n_select=50):
    '''
    Select candidate clusters for new round of simulations based on least counts policy.
    
    Args
    -------------
    kmeans (sklearn.cluster.KMeans): KMeans clustering object fitted on X_colvar.
    X (np.ndarray): array of shape (n_frames, D). Representative subset of frames from trajectories.
    X_colvar (np.ndarray): array of shape (n_frames, n_features). Representative subset of frames from trajectories.
    n_select (int), default 50: how many candidates to select based on least counts policy.
    
    Returns
    -------------
    candidate_indices (np.ndarray): array of shape (n_select,). Indices of clusters selected based on least counts.
    central_frames (np.ndarray): array of shape (n_select, D). Frames in X that are closest to the center of each candidate.
    central_frames_indices (np.ndarray): array of shape (n_select,). Indices of the frames in X that are closest to the center of each candidate.
    '''
    
    # Select n_select candidates via least counts
    counts = Counter(kmeans.labels_)
    least_counts = np.asarray(counts.most_common()[::-1][:n_select])[:,0] # Which clusters contain lowest populations
    candidate_indices = least_counts
    # Find frames closest to cluster centers of candidates
    least_counts_centers = kmeans.cluster_centers_[least_counts]
    central_frames_indices, _ = pairwise_distances_argmin_min(least_counts_centers, X_colvar)
    central_frames = X[central_frames_indices]
    
    return candidate_indices, central_frames, central_frames_indices

def compute_Vi_all_clusters(kmeans, X, d):
    '''
    Computes the matrices Vi. These matrices are of shape (D, d) (in this example notebook, (3, 1)) and contain the principal 
    component decomposition of each cluster. This function differs from compute_Vi in that it uses all clusters instead
    of the least count clusters only.
    
    Args
    -------------
    kmeans (sklearn.cluster.KMeans): KMeans clustering object fitted on X.
    X (np.ndarray): array of shape (n_frames, D). Representative subset of frames from trajectories.
    d (int): number of principal components to return.
    
    Returns
    -------------
    Vis (list[np.ndarrays]): list of Vi matrices. Order is given by kmeans cluster indices, which are arbitrary.
    '''
    Vis = []
    for i in range(kmeans.n_clusters):
        instances = np.where(kmeans.labels_ == i)
        points = X[instances]
        Vi = principal_components(points, d=d).T # Must transpose to get desired shape
        Vis.append(Vi)
    return Vis

def compute_Gi_all_clusters(central_frames):
    '''
    Computes the matrices Gi. These matrices are of shape (D, K) (in this example notebook, (3, 4)) and contain 
    the gradient of each collective variable w.r.t. to the cartesian coordinate of the point.
    The gradients are normalized according to the average norm.
    
    Args
    -------------
    central_frames (np.ndarray): array of shape (n_select, D). 
        Frames in that are closest to the center of each candidate.
    
    Returns
    -------------
    Gis (list[np.ndarrays]): list of Gi matrices. Order is identical to order of candidate_indices.
    '''
    Gis = []
    n = len(central_frames) # Same as n_select
    norm = 0
    for central_frame in central_frames:
        Gi = define_colvars_grad(central_frame)
        norm += np.linalg.norm(Gi, axis=0)
        Gis.append(Gi)
    
    # Scale each vector according to the average norm
    normed_Gis = []
    for Gi in Gis:
        normed_Gis.append(Gi*n/norm)
    
    return normed_Gis

def compute_agent_stakes(kmeans, agent_indices, n_agents, method='percentage', k=None):
    '''
    Returns agent stakes of a cluster given the number of frames from each agent that fall in said cluster.
    
    Args
    -------------
    kmeans (sklearn.cluster.KMeans): KMeans clustering object fitted on trajectory data or subsample.
    agent_indices (np.ndarray): array containing the agent indices for each frame.
    n_agents (int): number of agents.
    method (str): one of "percentage", "max", "equal", or "logistic". 
        percentage: stakes are proportional to number of frames that agents have in the cluster.
        max: the agent with the max number of frames in the cluster has all the stakes.
        equal: all agents with at least one frame in the cluster have the same stakes.
        logistic: percentage stakes are transformed as 1/(1+e^(-k(x-0.5))) and renormalized. Must set parameter k (higher k ~= max, lower k ~= equal).
    k (float): parameter for logistic method. Ignored if method is not set to logistic.
    
    Returns
    -------------
    agent_stakes (np.ndarray): array of shape (n_agents, n_clusters). Entry agent_stakes[i, j] indicates the stakes that agent i has on cluster j.
    '''
    if (method=="logistic"):
        try: 
            assert(k is not None)
        except AssertionError as err:
            raise AssertionError("k parameter must be set if using method='logistic'") from err
    
    stakes = np.zeros((n_agents, kmeans.n_clusters))
    for i in range(kmeans.n_clusters):
        stakes_cluster = np.zeros(n_agents)
        agent_idx = agent_indices[np.where(kmeans.labels_ == i)]
        agent_indices_count = Counter(agent_idx)
        total = sum(agent_indices_count.values())
        
        for agent, count in agent_indices_count.most_common():
            stakes_cluster[agent] = count/total
    
        if (method=="percentage"):
            stakes[:, i] = stakes_cluster

        elif (method=="logistic"):
            x0 = 0.5
            logistic_fun = lambda x: 1/(1 + np.exp(-k*(x-x0)))

            stakes_transformed = logistic_fun(stakes_cluster)
            stakes_transformed[np.where(stakes_cluster < 1e-18)] = 0 # Make sure that the function evaluates to 0 at x=0
            stakes_transformed /= stakes_transformed.sum() # Re-normalize
            stakes[:, i] = stakes_transformed

        elif (method=="equal"):
            stakes_cluster[np.where(stakes_cluster!=0)] = 1/np.count_nonzero(stakes_cluster)
            stakes[:, i] = stakes_cluster

        elif (method=="max"):
            # If two or more agents have the same number of frames (and these are equal to the maximum), one of them is picked arbitrarily to get stakes 1
            stakes_cluster[np.argmax(stakes_cluster)] = 1
            stakes_cluster[np.where(stakes_cluster < 1)] = 0
            stakes[:, i] = stakes_cluster
            
        else:
            raise ValueError("Method "+method+" not understood.") 

        try: 
            assert(np.abs(stakes[:, i].sum()-1) < 1e-6)
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
    X_a (np.ndarray): array of shape (?, D). Frames observed by a given agent.
        These are converted internally to colvar space.
    central_frames (np.ndarray): array of shape (n_select, D). Frames in X that are closest to the center of each candidate.
        These are converted internally to colvar space.
    stakes_a (np.ndarray): array of shape (n_select,). Stakes of the agent in each candidate.
    weights_a (np.ndarray): array of shape n_features,). Weights that the given agent assigns to each order parameter.
    n_select (int): how many candidates were selected based on least counts policy.
    
    Returns
    -------------
    rewards (np.ndarray): array of shape (n_select,). Rewards assigned to candidates by given agent.
    '''
    # Initialize array
    rewards = np.empty(n_select)
    
    # Convert frames to colvar space
    X_a_colvar = define_colvars(X_a)
    central_frames_colvar = define_colvars(central_frames)
    
    # Compute distribution parameters for frames observed by the agent
    mu = X_a_colvar.mean(axis=0)
    sigma = X_a_colvar.std(axis=0)
    
    # Compute distance of each candidate to the mean
    distances = (weights_a*np.abs(central_frames_colvar - mu)/sigma).sum(axis=1) # Shape is (n_select,)

    # Compute rewards
    # Since stakes_a is zero for those clusters that do not involve agent a, rewards are accurately set to zero here.
    rewards = stakes_a*distances
    
    return rewards

def get_weights(Vis, Gis, agent_stakes, which_agent):
    '''
    Derive weights from TSLC rewards function using agent stakes for the MA algorithm.
    
    Args
    -------------
    Vis (list[np.ndarrays]): list of Vi matrices. Shape of Vis[i] is (D, d).
    Gis (list[np.ndarrays]): list of Gi matrices. Shape of Gis[i] is (D, n_features).
    agent_stakes (np.ndarray): array of shape (n_agents, n_select). 
        Entry agent_stakes[i, j] indicates the stakes that agent i has on cluster j.
    which_agent (int): agent index for which weights are being computed.
    
    Returns
    -------------
    weights_a (np.ndarray): array of shape (n_features,). Weights that the given agent assigns to each order parameter.
    '''
    assert(len(Vis) == len(Gis) == len(agent_stakes[which_agent]))
    K = Gis[0].shape[1]
    
    A = np.zeros((K, K))
    
    for i, s in enumerate(agent_stakes[which_agent]):
        Gi = Gis[i]
        Vi = Vis[i]
        A += s*(Gi.T@Vi)@(Vi.T@Gi)
    
    eig_vals, eig_vecs = np.linalg.eig(A)
    idx = np.argmax(eig_vals)
    
    weights_a = eig_vecs[:, idx]**2
    
    return np.real_if_close(weights_a, tol=1e5)

def select_starting_points_MA_standard_Euclidean(X, central_frames, agent_idx, agent_stakes, weights, n_agents, n_chosen=10, collaborative=True, competitive=False):
    '''
    Select starting positions for new simulations. Here, the collective rewards are computed as the sum of the rewards from each agent.
    
    Args
    -------------
    X(np.ndarray): array of shape (n_frames, D). Representative subset of frames from trajectories.
    central_frames (np.ndarray): array of shape (n_select, D). Frames that are closest to the cluster center of each candidate.
    agent_idx (np.ndarray): array of shape (n_frames,) indicating which agent originated each frame.
    agent_stakes (np.ndarray): array of shape (n_agents, n_select). Entry agent_stakes[i, j] indicates the fraction of frames from 
        candidate j that were generated by agent i.
    weights (list[np.ndarray]): list of length n_agents with entries of shape (K,). Weights that each agent assigns to each order parameter.
    n_agents (int): number of agents.
    n_chosen (int): number of candidates that will be used for new trajectories.
    collaborative (Bool): if set to True (default), the rewards from all agents are added together. 
        If False, only the reward from the agent who returned the highest value is used.
    competitive (Bool): this option is only taken into account if collaborative is set to False. 
        If True, the reward is the difference between the non-collaborative reward minus the rewards assigned by all other agents.
    
    Returns
    -------------
    chosen_frames (np.ndarray): array of shape (n_chosen, D). Frames from which to launch new trajectories.
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
    chosen_frames (np.ndarray): array of shape (n_chosen, D). Frames from which to launch new trajectories.
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
            trajectories[0].append(traj) # Force all trajectories into the only agent
    
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
    
    Returns
    -------------
    None. Results are saved to output_dir.
    '''
    # Step 1: define some hyperparameters and initialize arrays --> To be provided via init_variables
    num_spawn = kwargs['num_spawn'] # Number of trajectories spawn per epoch
    n_select = kwargs['n_select'] # Number of least-count candidates selected per epoch
    n_agents = 1 # Forced to 1
    traj_len = kwargs['traj_len']
    n_features = kwargs['n_features'] # Equivalent to parameter K. Number of variables in OP space (in this case it's 4)
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
    weights = [np.ones((n_features))/n_features for _ in range(n_agents)] # The actual initial value doesn't matter here
    
    # Step 3: collect some initial data
    trajectories = [[]]
    trajectories = collect_initial_data(num_spawn, traj_len, potential, initial_positions, trajectories)
    
    # Steps 4-9: cluster, compute rewards, tune weights, run new simulations, and repeat

    # Logs
    least_counts_points_log = [] # For central frames from candidates
    agent_stakes_log = []
    weights_log = [[] for _ in range(n_agents)]
    individual_rewards_log = [[] for _ in range(n_agents)]
    selected_structures_log = []
    area_log = []
    area_agents_log = [[] for _ in range(n_agents)]
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for e in range(epochs):
        print("Running epoch: {}/{}".format(e+1, epochs), end='\n')

        # Clustering
        kmeans, X, X_colvar, agent_idx, Gis, Vis, agent_stakes = clustering_MA_FFT(trajectories, n_agents, n_select, stakes_method='percentage', stakes_k=10, max_frames=max_frames, b=b, gamma=gamma, d=d)

        # Select candidates
        candidate_indices, central_frames, central_frames_indices = select_least_counts_MA(kmeans, X, X_colvar, n_select=n_select)

        # Save logs
        least_counts_points_log.append(central_frames)
        agent_stakes_log.append(agent_stakes)
        
        # Compute rewards and weights (this step is done agent-wise)
        for a in range(n_agents):
            weights[a] = get_weights(Vis, Gis, agent_stakes, a)

            # Update logs
            weights_log[a].append(weights[a])
        
        agent_stakes_candidates = agent_stakes[:, candidate_indices]
        chosen_frames, executors = select_starting_points_MA_standard_Euclidean(X, central_frames, agent_idx, agent_stakes_candidates, weights, n_agents, n_chosen=num_spawn, collaborative=True, competitive=False)
        selected_structures_log.append(chosen_frames)

        trajectories = spawn_trajectories_MA(trajectories, chosen_frames, executors, potential=potential)
        
        ### Compute area explored ###
        concatenated_trajs = np.concatenate([trajectories[i][j] for i in range(n_agents) for j in range(len(trajectories[i]))])[:, :2]
        area_log.append(area_explored(potential_func, xlim, ylim, concatenated_trajs, threshold))
        color_exec = np.asarray(['black', 'white'])
        for a in range(n_agents):
            concatenated_trajs = np.concatenate([trajectories[a][j] for j in range(len(trajectories[a]))])[:, :2]
            area_agents_log[a].append(area_explored(potential_func, xlim, ylim, concatenated_trajs, threshold))
        
        if debug: 
            if (e % 20 == 0) or (e+1 == epochs):
                print('Weights:')
                for a in range(n_agents):
                    print(weights[a])
                print('Area explored:', area_log[-1])
                x_plot = np.arange(*xlim)
                y_plot = np.arange(*ylim)
                X_plot, Y_plot = np.meshgrid(x_plot, y_plot) # grid of point
                Z_plot = potential_func(X_plot, Y_plot) # evaluation of the function on the grid
                im = plt.imshow(Z_plot, cmap=plt.cm.jet, extent=[xlim[0], xlim[1], ylim[0], ylim[1]]) # drawing the function
                plt.colorbar(im) # adding the colobar on the right
                for a in range(n_agents):
                    plt.scatter(np.concatenate(trajectories[a])[::3,0], np.concatenate(trajectories[a])[::3,1], s=0.2, c='red')
                plt.scatter(central_frames[:,0], central_frames[:,1], s=20, alpha=0.5, c=agent_stakes_candidates[0])
                plt.scatter(chosen_frames[:,0], chosen_frames[:,1], s=1, alpha=1, c=color_exec[executors])
                plt.xlim([-3, 3])
                plt.ylim([-3, 3])
                plt.savefig(os.path.join(output_dir, output_prefix + 'landscape_epoch_{}.png'.format(e+1)), dpi=150)
                plt.close()
    
    ### Save results ###
    save_pickle(trajectories, os.path.join(output_dir, output_prefix + 'trajectories.pickle'))
    save_pickle(weights_log, os.path.join(output_dir, output_prefix + 'weights_log.pickle'))
    save_pickle(least_counts_points_log, os.path.join(output_dir, output_prefix + 'least_counts_points_log.pickle'))
    save_pickle(individual_rewards_log, os.path.join(output_dir, output_prefix + 'individual_rewards_log.pickle'))
    save_pickle(selected_structures_log, os.path.join(output_dir, output_prefix + 'selected_structures_log.pickle'))
    save_pickle(agent_stakes_log, os.path.join(output_dir, output_prefix + 'agent_stakes_log.pickle'))
    save_pickle(area_agents_log, os.path.join(output_dir, output_prefix + 'area_agents_log.pickle'))
    save_pickle(area_log, os.path.join(output_dir, output_prefix + 'area_log.pickle'))