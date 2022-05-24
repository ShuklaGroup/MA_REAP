import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from potentials import four_wells_asymmetric
from potentials import four_wells_asymmetric_func
from LeastCounts import run_trial

TASK_ID = int(sys.argv[1]) - 1 # From SGE (we must subtract 1 because it starts at 1 instead of 0). For parallel cluster run
trials_per_task = 1 # For parallel cluster run

kwargs = {
    'num_spawn': 20,  # Number of trajectories spawn per epoch
    'n_select': 20,  # Number of least-count candidates selected per epoch
    'traj_len': 500,  # Trajectory length
    'n_features': 2,  # Number of variables in OP space
    'd': 2,  # Parameter to determine number of clusters (intrinsic dimensionality given by potential function)
    'gamma': 0.6,  # Parameter to determine number of clusters
    'b': 3e-4,  # Parameter to determine number of clusters
    'max_frames': 1e5,  # Maximum # of frames to use
    'xlim': (-0.5, 2.5, 0.05),  # Limits where to compute area covered by agents
    'ylim': (-0.5, 2.5, 0.05),  # Limits where to compute area covered by agents
    'threshold': -20,  # Limits where to compute area covered by agents
    'debug': 1,  # I keep this on to save the images of the landscapes
    'potential_func': four_wells_asymmetric_func,  # Change depending on potential used
}

for i in range(trials_per_task):
    run_trial(
        potential=four_wells_asymmetric,  # Change according to desired potential
        initial_positions=[[0.2, 1, 0], [1.8, 1, 0]],
        # Change according to number of agents and their starting coordinates
        epochs=2,  # Change # of epochs as needed
        output_dir='testing',  # Change output directory as needed
        output_prefix='replicate_{}_'.format(i + TASK_ID * trials_per_task),  # Change output prefix as needed
        **kwargs
    )
