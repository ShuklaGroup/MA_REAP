'''
Settings used for trials presented in paper:

CITATION
'''

symmetric_cross = {
    'potential': four_wells_symmetric,
    'initial_positions': [[0.8, 1, 0], [1.2, 1, 0]],
    'epochs': 100,
    'num_spawn': 20,
    'n_select': 50,
    'n_agents': 2,
    'traj_len': 500,
    'delta': 0.02,
    'n_features': 2,
    'd': 2,
    'gamma': 0.6,
    'b': 3e-4,
    'max_frames': 1e5,
    'xlim': (-0.5, 2.5, 0.05),
    'ylim': (-0.5, 2.5, 0.05),
    'threshold': -20,
    'debug': 0,
    'potential_func': four_wells_symmetric_func,
    'stakes_method': 'percentage',
    'stakes_k': None,
    'collaborative': True,
    'competitive': False,
}

asymmetric_cross = {
    'potential': four_wells_asymmetric,
    'initial_positions': [[0.2, 1, 0], [1.8, 1, 0]],
    'epochs': 100,
    'num_spawn': 20,
    'n_select': 50,
    'n_agents': 2,
    'traj_len': 500,
    'delta': 0.02,
    'n_features': 2,
    'd': 2,
    'gamma': 0.6,
    'b': 3e-4,
    'max_frames': 1e5,
    'xlim': (-0.5, 2.5, 0.05),
    'ylim': (-0.5, 2.5, 0.05),
    'threshold': -20,
    'debug': 0,
    'potential_func': four_wells_asymmetric_func,
    'stakes_method': 'percentage',
    'stakes_k': None,
    'collaborative': True,
    'competitive': False,
}
