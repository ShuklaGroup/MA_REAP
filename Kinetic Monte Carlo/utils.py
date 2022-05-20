import pickle
import numpy as np
from matplotlib import pyplot as plt
import pyemma


def save_pickle(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)

def load_pickle(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)

def msm_from_mtx(transition_matrix, dt_model='5 ns'):
    '''
    Build an MSM from a sparse transition matrix. 
    
    Args
    -------------
    transition_matrix (scipy.sparse.csr_matrix): compressed sparse row transition matrix of shape (n_states, n_states).
    dt_model (str): default '5 ns' (used in Src example). Lag time for MSM.
    
    Returns
    -------------
    msm (pyemma.msm.models.msm.MSM): Markov state model. 
    '''
    # Note: dt_model='5 ns' (a.k.a. MSM lag time or tau) is based on 
    # specifications in the file https://uofi.app.box.com/file/499443522070 
    # (restricted accesss) by Zahra Shamsi
    P = np.asarray(transition_matrix.todense())
    return pyemma.msm.markov_model(P, dt_model=dt_model)

def run_monte_carlo_traj(msm, init_state=0, n_steps=50):
    '''
    Run a short Monte Carlo simulation starting on state `init_state` for `n_steps` under Markov state model `msm`.
    
    Args
    -------------
    msm (pyemma.msm.models.msm.MSM): Markov state model that will determine the kinetics of the simulation.
    init_state (int): initial state (default 0).
    n_steps (int): number of steps to run MC simulation (default 50). The step size is the lag time of the given MSM.
    
    Returns
    -------------
    trajectory (numpy.ndarray): trajectory of shape (n_steps,). The state trajectory.
    '''
    return msm.simulate(n_steps, start=init_state)

def map_state_to_coordinates(state_index_map, state_coordinates):
    '''
    Creates a mapping (callable) of a sequence of states to a sequence of points in collective variable-space.
    
    Args
    -------------
    sequence_states (np.ndarray[int]): array of shape (traj_len,). Sequence of discrete states from kinetic MC trajectory.
    state_index_map (np.ndarray[int]): array of shape (n_states_connected,). Array that maps from connected component of MSM to full MSM. 
        Let j = state_index_map[i], then j is the index of the state in the full MSM corresponding to the ith state in the 
        connected component of the MSM. If all the MSM components are in the connected component, then state_index_map[i] = i.
    state_coordinates (np.ndarray): array of shape (n_states_full, n_collective_variables). Array that maps an MSM state to its location
        in collective-variable space.
    
    Returns
    -------------
    sequence_coordinates (np.ndarray): array of shape (traj_len, n_collective_variables). Sequence of points in collective varaible space.
    '''
    
    def mapping(sequence_states):
        full_msm_indexes = state_index_map[sequence_states]
        sequence_coordinates = state_coordinates[full_msm_indexes]
        
        return sequence_coordinates
    
    return mapping

def num_clusters(sequence_coordinates):
    '''
    Use some heuristic to compute the number of clusters for KMeans.
    
    For kinetic MC simulations, the number of clusters will equal the number of states discovered.
    '''
    return len(np.unique(sequence_coordinates))

def spawn_trajectories_MA(state_sequences, msm, chosen_frames, executors, seed=None, traj_len=500, potential=''):
    '''
    Spawns new trajectories and assigns them to the correct agent.
    
    Args
    -------------
    state_sequences (list[list[np.ndarray]]): trajectories collected so far (in terms of MSM states). 
        They should be accessed as state_sequences[ith_agent][jth_trajectory].
    chosen_frames_indices (np.ndarray): array of shape (n_chosen,). Frame indices from which to launch new trajectories.
    executors (np.ndarray): array of shape (n_chosen,). Index of the agent that launches each trajectory.
    
    Returns
    -------------
    state_sequences (list[list[np.ndarray]]): trajectories collected so far (in terms of MSM states).
    '''
    
    for init_state, agent in zip(chosen_frames, executors):
        new_traj = run_monte_carlo_traj(msm, init_state=init_state, n_steps=traj_len)
        state_sequences[agent].append(new_traj)
        
    return state_sequences