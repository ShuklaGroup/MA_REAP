import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from potentials import circular_potential, circular_potential_func
from SA_TSLC import run_trial

TASK_ID = int(sys.argv[1]) - 1 # From SGE (we must subtract 1 because it starts at 1 instead of 0) --> for parallel runs in clusters
trials_per_task = 1 # For parallel runs in clusters

# Define some parameters for visualization
xlim = (-4, 4, 0.01)
ylim = xlim
def circular_potential_func_plot(x, y):
    return circular_potential_func((x, y, 0), c=-250, a=-10)

kwargs = {
    'num_spawn': 4, # Number of trajectories spawn per epoch
    'n_select': 50, # Number of least-count candidates selected per epoch
    'n_agents': 1, # Set to 1 anyway
    'traj_len': 500, # Trajectory length
    'n_features': 4, # Number of variables in OP space
    'd': 1, # Parameter to determine number of clusters (intrinsic dimensionality given by potential function)
    'gamma': 0.7, # Parameter to determine number of clusters
    'b': 7e-3, # Parameter to determine number of clusters
    'max_frames': 5e5, # Maximum # of frames to use
    'xlim': (-2.5, 2.5, 0.05), # Limits where to compute area covered by agents
    'ylim': (-2.5, 2.5, 0.05), # Limits where to compute area covered by agents
    'threshold': -247, # Limits where to compute area covered by agents
    'debug': 1, # Set to 1 to get plots of the landscape after a certain number of iterations
    'potential_func': circular_potential_func_plot, # Change depending on potential used
    'stakes_method': 'percentage', # Ignored
    'stakes_k': None, # Ignored
    'collaborative': True, # Ignored
    'competitive': False, # Ignored
}

# Define starting points
radius = 2
angle_1 = np.pi*3/4
angle_2 = np.pi*1/4
start_1 = [radius*np.cos(angle_1), radius*np.sin(angle_1), 0]
start_2 = [radius*np.cos(angle_2), radius*np.sin(angle_2), 0]

for i in range(trials_per_task):
    run_trial(
        potential = circular_potential, # Change according to desired potential
        initial_positions = [start_1, start_2], # Assigned to the only agent
        epochs = 200, # Change # of epochs as needed
        output_dir = 'testing', # Change output directory as needed
        output_prefix = 'replicate_{}_'.format(i + TASK_ID*trials_per_task), # Change output prefix as needed
        **kwargs
    )