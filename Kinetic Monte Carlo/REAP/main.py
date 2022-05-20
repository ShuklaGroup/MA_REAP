import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.io import mmread
from utils import msm_from_mtx, map_state_to_coordinates, load_pickle
from SA_REAP import run_trial

###############################  Src Kinase example  ###############################

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'example_data')
cv1 = np.load(os.path.join(data_dir, 'Src', 'Gens_aloopRMSD.npy'))
cv2 = np.load(os.path.join(data_dir, 'Src', 'Gens_y_KE.npy'))
state_coordinates = np.vstack([cv1, cv2]).T 
state_index_map = np.load(os.path.join(data_dir, 'Src', 'map_Macro2micro.npy'))

mapping = map_state_to_coordinates(state_index_map, state_coordinates)
msm = msm_from_mtx(mmread(os.path.join(data_dir, 'Src', 'tProb.mtx')))

# Run trials
n_trials = 2 # Change as desired
epochs = 20 # Change as desired

kwargs = {
    'num_spawn': 2, # Number of trajectories spawn per epoch
    'n_select': 12, # Number of least-count candidates selected per epoch
    'n_agents': 1, # Forced to 1
    'traj_len': 10, # The real # of transitions per trajectory is traj_len-1
    'delta': 0.1, # Upper boundary for learning step
    'n_features': 2, # Number of variables in OP space
    'max_frames': 1e4, # Max # of frames to sample from total data
    'mapping': mapping, # Maps MSM microstate to vector in CV-space
    'stakes_method': 'max', # Stakes function
    'stakes_k': None, # Stakes k is ignored
}

A = 0 # As indexed in MSM
B = 8 # As indexed in MSM
initial_states = [A, B]

output_dir = 'testing' # Change as desired
reset = 0 # Only used for repeated trials

for t in range(reset, n_trials):
    run_trial(msm, 
              initial_states, 
              epochs, 
              output_dir=output_dir, 
              output_prefix='trial_{}_'.format(t),
              **kwargs)

####################################################################################

################################  OsSWEET2b example ################################

# data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'example_data')
# cv1 = np.load(os.path.join(data_dir, 'OsSWEET2b', 'periplasmic_dist.npy'))
# cv2 = np.load(os.path.join(data_dir, 'OsSWEET2b', 'cytoplasmic_dist.npy'))
# state_coordinates = np.vstack([cv1, cv2]).T
# state_index_map = np.arange(900)

# mapping = map_state_to_coordinates(state_index_map, state_coordinates)

# msm = load_pickle(os.path.join(data_dir, 'OsSWEET2b', 'MSM-osSWEET-holo-900cls-3ticks.pkl'))

# n_trials = 2 # Change as desired
# epochs = 20 # Change as desired

# kwargs = {
#     'num_spawn': 2, # Number of trajectories spawn per epoch
#     'n_select': 12, # Number of least-count candidates selected per epoch
#     'n_agents': 1, # Forced to 1
#     'traj_len': 2, # The real # of transitions per trajectory is traj_len-1
#     'delta': 0.01, # Upper boundary for learning step
#     'n_features': 2, # Number of variables in OP space
#     'max_frames': 1e4, # Max # of frames to sample from total data
#     'mapping': mapping, # Maps MSM microstate to vector in CV-space
#     'stakes_method': 'max', # Stakes function
#     'stakes_k': None, # Stakes k is ignored
# }

# OF = 508
# OC = 262
# IF = 673

# initial_states = [IF, OF]

# output_dir = 'testing' # Change as desired
# reset = 0

# for t in range(reset, n_trials):
#     run_trial(msm, 
#               initial_states, 
#               epochs, 
#               output_dir=output_dir, 
#               output_prefix='trial_{}_'.format(t),
#               **kwargs)

####################################################################################