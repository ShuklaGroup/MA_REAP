import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MA_REAP import run_trial

n_trials = 2  # Number of trials

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'example_data')

kwargs = {
    'num_spawn': 4,  # Number of trajectories spawn per epoch
    'n_select': 12,  # Number of least-count candidates selected per epoch
    'n_agents': 2,
    'traj_len': 1000,  # Change as needed
    'delta': 0.02,  # Upper boundary for learning step
    'n_features': 2,  # Number of variables in OP space
    'd': 2,  # Parameter to determine number of clusters (intrinsic dimensionality given by potential function)
    'gamma': 0.7,  # Parameter to determine number of clusters
    'b': 2e-3,  # Parameter to determine number of clusters
    'max_frames': 1e4,  # Max number of frames to use for clustering
    'stakes_method': 'percentage',
    'stakes_k': None,
    'topology': os.path.join(data_dir, 'alanine_dipeptide.pdb'),  # Topology file
}

restart = 0  # Restart at a given iteration
initial_structures = [os.path.join(data_dir, 'starting_1.pdb'),
                      os.path.join(data_dir, 'starting_2.pdb')]  # Provide initial structures

for r in range(restart, n_trials):
    print('Running trial {}/{}'.format(r + 1, n_trials))

    run_trial(
        initial_structures=initial_structures,
        epochs=11,
        output_dir=os.path.join('run_trials', 'trial_{}'.format(r + 1)),
        output_prefix='',
        **kwargs
    )
