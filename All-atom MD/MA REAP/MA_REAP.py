import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import openmm as mm
import openmm.app as app
from simtk.unit import *
import mdtraj as md
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from utils import save_pickle, load_pickle, frame_to_openmm

def clustering_MA(trajectories, trajectories_log, n_agents, n_select, n_clusters=None, max_frames=1e5, b=1e-4, gamma=0.7, d=2):
    '''
    Clusters all (or a representative subset) of the frames in trajectories using KMeans. Returns clustering object, which will be used to select the 
    least counts clusters. The agents that created each frame are remembered and their indices are returned as well. The selected subset of the total
    frames are also returned.
    
    Args
    -------------
    trajectories (list[list[np.ndarray]]): trajectories collected so far. They should be accessed as trajectories[ith_agent][jth_trajectory].
    trajectories_log (list[list[str]]): list of filenames containing each trajectory.
        They should be accessed as trajectories_log[ith_agent][jth_trajectory].
    n_agents (int): number of agents.
    n_clusters (int), default None: number of clusters to use for KMeans. If None, a heuristic is used to approximate the number of clusters needed.
    max_frames (int), default 1e5: maximum number of frames to use in the clustering step. If set to 0 or None, all frames are used.
    b (float), default 1e-4: coefficient for n_clusters heuristic.
    gamma (float), default 0.7: exponent for n_clusters heuristic (should theoretically be in (0.5, 1)).
    d (float), default 2: intrinsic dimensionality of slow manifold for the system. Used in n_clusters heuristic.
    
    Returns
    -------------
    KMeans (sklearn.cluster.KMeans): fitted KMeans clustering object.
    X (np.ndarray): array of shape (max_frames, n_features) containing the subset of the data used for clustering.
    agent_idx (np.ndarray): array of shape (max_frames,) containing the index of the agent that originated each frame.
    file_names (np.ndarray[str]): array of shape (max_frames,) containing the filename where each frame in X is stored.
    frame_indices (np.ndarray): array of shape (max_frames,) containing the indices of the frames for the corresponding trajectory file.
    '''
    
    # Put frames in format that is usable for KMeans
    assert(n_agents == len(trajectories))
    total_frames = 0
    trajectory = [] # All frames
    agent_index = [] # Array mapping a frame index in trajectory to its corresponding agent
    file_names = [] # Array mapping frames to trajectory files
    frame_indices = [] # List mapping frames to their index in the trajectory file
    for a, agent_trajs in enumerate(trajectories):
        trajs_filenames = trajectories_log[a]
        for i, traj in enumerate(agent_trajs):
            total_frames += len(traj)
            trajectory.append(traj)
            agent_index.extend([a]*len(traj))
            file_names.extend([trajs_filenames[i]]*len(traj))
            frame_indices.extend(list(range(len(traj))))
          
    trajectory = np.concatenate(trajectory)
    agent_index = np.asarray(agent_index, dtype=int)
    file_names_full = np.asarray(file_names, dtype=object)
    frame_indices_full = np.asarray(frame_indices, dtype=int)
    
    # Downsample number of points
    if (not max_frames) or (total_frames <= max_frames):
        X = trajectory
        agent_idx = agent_index
        file_names = file_names_full
        frame_indices = frame_indices_full
    
    elif total_frames > max_frames:
        max_frames = int(max_frames)
        rng = np.random.default_rng()
        rand_indices = rng.choice(len(trajectory), max_frames, replace=False)
        X = trajectory[rand_indices]
        agent_idx = agent_index[rand_indices]
        file_names = file_names_full[rand_indices]
        frame_indices = frame_indices_full[rand_indices]
                                  
#     print(X.shape)
    # Use heuristic from https://openreview.net/pdf?id=00thAjcutwh to determine number of clusters
    if (n_clusters is None):
        n_clusters = int(b*(min(total_frames, max_frames)**(gamma*d)))
    if (n_clusters < n_select):
        n_clusters = n_select
    kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=5).fit(X)
    
    return kmeans, X, agent_idx, file_names, frame_indices

def select_least_counts_MA(kmeans, X, agent_idx, n_agents, stakes_method, stakes_k=None, n_select=50):
    '''
    Select candidate clusters for new round of simulations based on least counts policy.
    
    Args
    -------------
    kmeans (sklearn.cluster.KMeans): KMeans clustering object fitted on X.
    X (np.ndarray): array of shape (n_frames, n_features). Representative subset of frames from trajectories.
    agent_idx (np.ndarray): array of shape (n_frames,) indicating which agent originated each frame.
    n_agents (int): number of agents.
    n_select (int), default 50: how many candidates to select based on least counts policy.
    
    Returns
    -------------
    central_frames (np.ndarray): array of shape (n_select, n_features). Frames in X that are closest to the center of each candidate.
    central_frames_indices (np.ndarray): array of shape (n_select,). Indices of the frames in X that are closest to the center of each candidate.
    agent_stakes (np.ndarray): array of shape (n_agents, n_select). Entry agent_stakes[i, j] indicates the fraction of frames from 
        candidate j that were generated by agent i.
    '''
    
    # Select n_select candidates via least counts
    counts = Counter(kmeans.labels_)
    least_counts = np.asarray(counts.most_common()[::-1][:n_select])[:,0] # Which clusters contain lowest populations
    
    # Find frames closest to cluster centers of candidates
    least_counts_centers = kmeans.cluster_centers_[least_counts]
    central_frames_indices, _ = pairwise_distances_argmin_min(least_counts_centers, X)
    central_frames = X[central_frames_indices]
    
    # Compute agent stakes
    agent_stakes_raw = np.zeros((n_agents, n_select))
    for candidate_idx, candidate in enumerate(least_counts):
        agent_indices = agent_idx[np.where(kmeans.labels_ == candidate)]
        total = len(agent_indices)
        agent_indices_count = Counter(agent_indices).most_common()
        for agent, count in agent_indices_count:
            agent_stakes_raw[agent, candidate_idx] = count/total

    agent_stakes = compute_agent_stakes(agent_stakes_raw, method=stakes_method, k=stakes_k)
        
    return central_frames, central_frames_indices, agent_stakes

def compute_agent_stakes(agent_stakes_raw, method='percentage', k=None):
    '''
    Returns agent stakes of a cluster given the number of frames from each agent that fall in said cluster.
    
    Args
    -------------
    agent_stakes_raw (np.ndarray): array of shape (n_agents, n_clusters). Stakes if method used is percentage. 
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
    
    if method == 'percentage':
        return agent_stakes_raw
    
    elif method == 'max':
        agent_stakes = np.empty(agent_stakes_raw.shape)
        for n in range(agent_stakes_raw.shape[1]):
            agent_stakes[:, n] = (agent_stakes_raw[:, n] == agent_stakes_raw[:, n].max()).astype(int)
        return agent_stakes
    
    elif method == 'equal':
        agent_stakes = np.empty(agent_stakes_raw.shape)
        for n in range(agent_stakes_raw.shape[1]):
            agent_stakes[:, n][np.where(agent_stakes_raw[:, n] != 0)] = 1/np.count_nonzero(agent_stakes_raw[:, n])
        return agent_stakes
    
    elif method == 'logistic':
        if k is None:
            raise ValueError('k must be specified if using method logistic')
        
        x0 = 0.5
        logistic_fun = lambda x: 1/(1 + np.exp(-k*(x-x0)))
        
        agent_stakes = np.empty(agent_stakes_raw.shape)
        for n in range(agent_stakes_raw.shape[1]):
            stakes_transformed = logistic_fun(agent_stakes_raw[:, n])
            stakes_transformed[np.where(agent_stakes_raw[:, n] < 1e-18)] = 0 # Make sure that the function evaluates to 0 at x=0
            stakes_transformed /= stakes_transformed.sum() # Re-normalize
            agent_stakes[:, n] = stakes_transformed
            
        return agent_stakes
    
    else:
        raise ValueError("Method "+method+" not understood. Must choose 'percentage', 'max', 'equal', or 'logistic'.")

def compute_cumulative_reward_MA_standard_Euclidean(X, agent_idx, agent_stakes, central_frames_indices, n_select, n_agents, weights, which_agent):
    '''
    Returns the cumulative reward for current weights and a callable to the cumulative reward function (necessary to finetune weights).
    Note that this is the cumulative reward and reward function for the given agent.
    In the multi-agent implementation, this function is called n_agents times during the optimization step.
    
    Args
    -------------
    X (np.ndarray): array of shape (n_frames, n_features). Representative subset of frames from trajectories.
    agent_idx (np.ndarray): array of shape (n_frames,) indicating which agent originated each frame.
    agent_stakes (np.ndarray): array of shape (n_agents, n_select). Entry agent_stakes[i, j] indicates the fraction of frames from 
        candidate j that were generated by agent i.
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
    mu = X_a.mean(axis=0)
    sigma = X_a.std(axis=0)
    
    # Compute distance of each candidate to the mean
    distances = (weights_a*np.abs(central_frames - mu)/sigma).sum(axis=1) # Shape is (n_select,)
    
    # Compute rewards
    # Since stakes_a is zero for those clusters that do not involve agent a, rewards are accurately set to zero here.
    rewards = stakes_a*distances
    
    return rewards

def tune_weights_MA_standard_Euclidean(rewards_function, weights_a, delta=0.02):
    '''
    Defines constraints for optimization and maximizes rewards function. Returns OptimizeResult object.
    This function is called once per each agent per epoch.
    
    Args
    -------------
    rewards_function (callable): reward function to maximize. This corresponds to a given agent.
    weights_a (np.ndarray): array of shape (n_features,). Weights that the given agent assigns to each order parameter.
    delta (float): maximum amount by which an entry in the weights matrix can change. Think of it as an upper-bounded learning rate.
    
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

def select_starting_points_MA_standard_Euclidean(X, central_frames, agent_idx, agent_stakes, weights, n_agents, n_chosen=10):
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
    
    Returns
    -------------
    chosen_frames_indices (np.ndarray): array of shape (n_chosen,). Indices for frames in central_frames from which to launch new trajectories.
    chosen_frames (np.ndarray): array of shape (n_chosen, n_features). Frames from which to launch new trajectories.
    executors (np.ndarray): array of shape (n_chosen,). Index of the agent that launches each trajectory.
    '''
    
    # Compute collective rewards for each candidate with the updated weights
    n_select = len(central_frames)
    rewards = np.zeros(n_select)
    
    for a in range(n_agents):
        # Acess data for specific agent
        weights_a = weights[a]
        stakes_a = agent_stakes[a]

        indices = np.where(agent_idx == a)
        X_a = X[indices]
        rewards += compute_structure_reward_MA_standard_Euclidean(X_a, central_frames, stakes_a, weights_a, n_select)
    
    chosen_frames_indices = np.argsort(rewards)[-n_chosen:][::-1] # Indices of frames with maximum reward
    chosen_frames = central_frames[chosen_frames_indices] # Frames that will be used to start new simulations
    executors = agent_stakes.argmax(axis=0)[chosen_frames_indices] # Agents that will run the new trajectories 
    
    return chosen_frames_indices, chosen_frames, executors

#########################  Functions specific to alanine dipeptide #########################

def alanine_dipeptide_system(top_file):
    '''
    Returns alanine dieptide system in explicit solvent for simulation with OpenMM.
    Starting structure and simulation parameters are taken from original REAP paper.
    
    Args
    -------------
    top_file (str): path to topology.
    
    Returns
    -------------
    system (simtk.openmm.System): system object corresponding to alanine dipeptide.
    topology (simtk.openmm.app.Topology): alanine dipeptide topology.
    integrator (simtk.openmm.Integrator): integrator object to run simulation.
    positions (simtk.unit.Quantity[simtk.unit.Vec3]): initial atom positions.
    '''
    pdb = app.PDBFile(top_file)
    forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
    system = forcefield.createSystem(pdb.topology, 
                                     nonbondedMethod=app.PME,
                                     nonbondedCutoff=1*nanometer, 
                                     constraints=app.HBonds)
    integrator = mm.LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    system.addForce(mm.MonteCarloBarostat(1*bar, 300*kelvin))
    
    return system, pdb.topology, integrator, pdb.positions 

def project_dihedral_alanine_dipeptide(traj_name, top_file):
    '''
    Projects a trajectory of alanine dipeptide into a 2d dihedral angle space.
    
     Args
    -------------
    top_file (str): path to topology (for alanine dipeptide system).
    traj_name (str): name of trajectory to project. Format must be compatible with mdtraj.
    
    Returns
    -------------
    data (numpy.ndarray): array of shape (n_frames, 2). 
    '''
    traj = md.load(traj_name, top=top_file)
    
    indices_phi, phi = md.compute_phi(traj)
    indices_psi, psi = md.compute_psi(traj)
    
    assert(phi.shape[1] == 1)
    assert(psi.shape[1] == 1)
    
    data = np.hstack((phi, psi))
    
    assert(psi.shape[0] == data.shape[0])
    assert(data.shape[1] == 2)
    
    return data

############################################################################################

def run_trajectory(system, topology, integrator, positions, output_file, n_steps=1000, platform='CUDA'):
    '''
    Run a trajectory of the specified system starting from the specified position.
    
    Args
    -------------
    system (simtk.openmm.System): system object.
    topology (simtk.openmm.app.Topology): alanine dipeptide topology.
    positions (simtk.unit.Quantity): initial atom positions.
    output_file (str): name of .dcd trajectory that will be saved.
    n_steps (int): number of steps to run. 
    platform (str): name of platform where OpenMM will run the simulation.
    
    Returns
    -------------
    None
    '''
    mm.Platform.getPlatformByName(platform)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.reporters.append(app.DCDReporter(output_file, 100))
    simulation.step(n_steps)

def spawn_trajectories(initial_positions, top_file, n_steps=1000, output_files=[]):
    '''
    Spawn trajectories from the chosen frames.
    
    Args
    -------------
    initial_positions (list[simtk.unit.Quantity[simtk.unit.Vec3]]): list of initial positions to start new simulations.
    output_files (list[str]): name of each trajectory to be saved. Recommended format 'agent_{agent#}_round_{round#}_traj_{traj#}.dcd'. The dcd format is hard-coded.
    
    Returns
    -------------
    None
    '''
    for positions, output_file in zip(initial_positions, output_files):
        system, topology, integrator, _ =  alanine_dipeptide_system(top_file)
        run_trajectory(system, topology, integrator, positions, output_file, n_steps=n_steps)

def collect_initial_data(num_trajectories, traj_len, initial_positions, output_dir, output_prefix, top_file):
	'''
	Collect some initial data before using adaptive sampling.

	Args
    -------------
    num_trajectories (int): number of trajectories to launch per agent.
    traj_len (int): number of simlated steps per trajectory.
    initial_positions (list[simtk.unit.Quantity[simtk.unit.Vec3]]): list of initial positions to start new simulations.
    output_dir (str): directory to save trial data.
    output_prefix (str): prefix for trial data.
    
    Returns
    -------------
    trajectories (list[list[np.ndarray]]): trajectories (in CV-space). The 'shape' is (n_agents, num_trajectories, traj_len, n_features).
    trajectories_log (list[list[str]]): list of filenames containing each trajectory.
        They should be accessed as trajectories_log[ith_agent][jth_trajectory].
	'''

	# Create list of output files
	basename = os.path.join(output_dir, output_prefix + 'agent_{}_init_traj_{}.dcd')
	n_agents = len(initial_positions)
	output_files = [ basename.format(a, t) for a in range(n_agents) for t in range(num_trajectories) ]
	print(output_files)

	# Create array of initial positions for spawn trajectories
	init_pos = [initial_positions[a] for a in range(n_agents) for _ in range(num_trajectories)]

	spawn_trajectories(init_pos, top_file, n_steps=traj_len, output_files=output_files)


	# Create initial trajectory log
	trajectories_log = [ [ basename.format(a, t) for t in range(num_trajectories) ]  for a in range(n_agents) ]
	print(trajectories_log)

	trajectories = [ [] for _ in range(n_agents) ]
	for a in range(n_agents):
		for t in range(num_trajectories):
			fname = basename.format(a, t)
			data = project_dihedral_alanine_dipeptide(fname, top_file)
			trajectories[a].append(data)

	return trajectories, trajectories_log

def run_trial(initial_structures, epochs, output_dir='', output_prefix='', **kwargs):
    '''
    Runs a trial of MA REAP with standard Euclidean distance rewards.
    
    Args
    -------------
    initial_structures (list[str]): path to starting structures for simulations. Lenght of list must match number of agents.
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
        topology (str): topology file used to create OpenMM system. Will be used to project coordinates into CVs as well setup simulations.
        stakes_method (str): method used to compute agent stakes.
        stakes_k (float): k parameter for logistic stakes.
    
    Returns
    -------------
    None. Results are saved to output_dir.
    '''

    # Create directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: define some hyperparameters and initialize arrays --> To be provided via init_variables
    num_spawn = kwargs['num_spawn'] # Number of trajectories spawn per epoch
    n_select = kwargs['n_select'] # Number of least-count candidates selected per epoch
    n_agents = kwargs['n_agents']
    traj_len = kwargs['traj_len']
    delta = kwargs['delta'] # Upper boundary for learning step
    n_features = kwargs['n_features']
    d = kwargs['d'] # Parameter to determine number of clusters (intrinsic dimensionality given by potential function)
    gamma = kwargs['gamma'] # Parameter to determine number of clusters
    b = kwargs['b'] # Parameter to determine number of clusters
    max_frames = kwargs['max_frames']
    top_file = kwargs['topology']
    stakes_method = kwargs['stakes_method']
    stakes_k = kwargs['stakes_k']
    
    # Step 2: set initial weights
    weights = [np.ones((n_features))/n_features for _ in range(n_agents)]
    
    # Step 3: collect some initial data
    #Convert initial structures to OpenMM positions
    assert(len(initial_structures) == n_agents)
    initial_positions = []
    for fname in initial_structures:
        xyz = md.load(fname).xyz[0]
        initial_positions.append(frame_to_openmm(xyz))
    trajectories, trajectories_log = collect_initial_data(num_spawn, traj_len, initial_positions, output_dir, output_prefix, top_file)
    
    # Logs
    least_counts_points_log = [] # For central frames from candidates
    agent_stakes_log = []
    cumulative_reward_log = [[] for _ in range(n_agents)]
    weights_log = [[] for _ in range(n_agents)]
    individual_rewards_log = [[] for _ in range(n_agents)]
    selected_structures_log = []
    executors_log = []
    
    for e in range(epochs):
        print("Running epoch: {}/{}".format(e+1, epochs), end='\r')

        # Clustering
        kmeans, X, agent_idx, file_names, frame_indices = clustering_MA(trajectories, trajectories_log, n_agents, n_select, max_frames=max_frames, b=b, gamma=gamma, d=d)

        # Select candidates
        central_frames, central_frames_indices, agent_stakes = select_least_counts_MA(kmeans, X, agent_idx, n_agents, stakes_method, stakes_k=stakes_k, n_select=n_select)

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

            # Save logs
            cumulative_reward_log[a].append(R)
            weights_log[a].append(weights[a])
            individual_rewards_log[a].append(individual_rewards)

        chosen_frames_indices, chosen_frames, executors = select_starting_points_MA_standard_Euclidean(X, central_frames, agent_idx, agent_stakes, weights, n_agents, n_chosen=num_spawn)
        
        selected_structures_log.append(chosen_frames)
        executors_log.append(executors)
        
        # Convert chosen frames to OpenMM positions
        starting_positions = []
        starting_filenames = file_names[central_frames_indices][chosen_frames_indices]
        starting_file_frames_indices = frame_indices[central_frames_indices][chosen_frames_indices]
        for fname, frame_idx in zip(starting_filenames, starting_file_frames_indices):
            xyz = md.load(fname, top=top_file).xyz[frame_idx]
            starting_positions.append(frame_to_openmm(xyz))
    
        # Create output names for trajectories
        output_filenames = []
        round_num = str(e+1) # Epoch starting in 1
        count_trajs = [1, 1] # To name each trajectory for each agent using sequential numbers
        for ex in executors:
            fname = os.path.join(output_dir, output_prefix, 'agent_{}_round_{}_traj_{}.dcd'.format(str(ex), round_num, str(count_trajs[ex])))
            count_trajs[ex] += 1
            trajectories_log[ex].append(fname)
            output_filenames.append(fname)

        # Run trajectories
        spawn_trajectories(starting_positions, top_file, n_steps=traj_len, output_files=output_filenames)

        # Project data to use in next round
        for fname, ex in zip(output_filenames, executors):
            data = project_dihedral_alanine_dipeptide(fname, top_file)
            trajectories[ex].append(data)

        if (e % 10 == 0):
            print('Weights:', weights)
            plt.scatter(np.concatenate(trajectories[0])[::3,0], np.concatenate(trajectories[0])[::3,1], s=0.5)
            plt.scatter(np.concatenate(trajectories[1])[::3,0], np.concatenate(trajectories[1])[::3,1], s=0.5)
            plt.scatter(chosen_frames[:,0], chosen_frames[:,1], s=10, c='r')
            plt.xlim([-np.pi, np.pi])
            plt.ylim([-np.pi, np.pi])
            plt.savefig(os.path.join(output_dir, output_prefix + 'landscape_epoch_{}.png'.format(e)), dpi=150)
            plt.close()
            
    ### Save results ###
    save_pickle(trajectories, os.path.join(output_dir, output_prefix + 'trajectories.pickle'))
    save_pickle(weights_log, os.path.join(output_dir, output_prefix + 'weights_log.pickle'))
    save_pickle(least_counts_points_log, os.path.join(output_dir, output_prefix + 'least_counts_points_log.pickle'))
    save_pickle(cumulative_reward_log, os.path.join(output_dir, output_prefix + 'cumulative_reward_log.pickle'))
    save_pickle(individual_rewards_log, os.path.join(output_dir, output_prefix + 'individual_rewards_log.pickle'))
    save_pickle(selected_structures_log, os.path.join(output_dir, output_prefix + 'selected_structures_log.pickle'))
    save_pickle(agent_stakes_log, os.path.join(output_dir, output_prefix + 'agent_stakes_log.pickle'))
    save_pickle(executors_log, os.path.join(output_dir, output_prefix + 'executors_log.pickle'))