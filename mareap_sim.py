"""
This script produces input files to perform MA REAP simulations with any MD engine (the file formats must be recognized
by mdtraj).
"""

import sys
import os
import argparse
import pickle
from glob import glob
from collections import Counter
import numpy as np
from natsort import natsorted
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import normalize
from scipy.optimize import minimize
import mdtraj as md


def set_parser():
    """
    Construct parser for mareap_sim.py.
    :return: dict with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python mareap_sim.py",
        description="""
        This script produces input files to perform MA REAP simulations with an MD engine of your choice.
        Run in a directory where data is organized as follows:
        {working directory}
            |_agent_1
            |   |__round_1
            |   |  |__traj_1
            |   |  |   |__{name of traj file}.npy           --> This is the featurized trajectory
            |   |  |   |__{name of traj file}.{format_traj} --> This is the original trajectory file
            |   |  |__traj_2
            |   |  |...
            |   |  |__traj_{N}
            |   |__round_2
            |   |...
            |   |__round_{M}
            |_agent_2
            |...
            |_agent_{P}
        
        The script will create a new directory, `round_{M+1}` that will contain subdirectories with input files for each trajectory.
        """,
        epilog="For details on MA REAP, see Kleiman et al. J. Chem. Theory Comput. 2022, 18, 9, 5422–5434.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('-r', '--reset', help="Path to MAREAP reset file (.pkl). This file is created when running the "
                                              "script for the first time. Some settings are ignored if passing a "
                                              "reset file.")
    parser.add_argument('-t', '--topology', help="Path to the topology file. (Ignored if passing reset file.)")
    parser.add_argument('-ft', '--format_traj', help="Format suffix for trajectory files. (Ignored if passing reset "
                                                     "file.)")
    parser.add_argument('-fc', '--format_coor', help="Format suffix for new input files.")
    parser.add_argument('-fs', '--frame_stride', help="Frame stride used when featurizing trajectories. (Ignored if "
                                                      "passing reset file.)", default=1,
                        type=int)
    parser.add_argument('-d', '--delta', help="Delta parameter for MA REAP. Default: 0.05. (Ignored if passing reset "
                                              "file.)", default=0.05, type=float)
    parser.add_argument('-w', '--weights', help="Initial feature weights for MA REAP (list of floats must add to 1). "
                                                "(Ignored if passing reset file.)",
                        nargs='+', type=float)
    parser.add_argument('-n', '--n_candidates', help="Number of least count candidates to consider.", type=int)
    parser.add_argument('-o', '--n_output', help="Number of trajectories for next round of simulations.", type=int)
    parser.add_argument('-c', '--clusters', help="Number of clusters to use (KMeans).", type=int)
    parser.add_argument('-s', '--stakes_method', help="Method to calculate stakes. Default: percentage. (Ignored if "
                                                      "passing reset file.)",
                        choices=['percentage', 'equal', 'logistic'], default='percentage')
    parser.add_argument('-sk', '--stakes_k', help="k parameter when using logistic stakes. Only needs to be set when "
                                                  "using -s logistic. (Ignored if passing reset file.)", type=float)
    parser.add_argument('-reg', '--regime', help="Regime used to combine rewards from different agents. Default: "
                                                 "collaborative. (Ignored if passing reset file.)",
                        default='collaborative', choices=['collaborative', 'noncollaborative', 'competitive'])
    return parser.parse_args()


def read_reset_file(filepath):
    """
    Load reset file.

    :param filepath: str.
        Path to reset file.
    :return: dict.
        Dictionary with settings.
    """
    if os.path.exists(filepath):
        print(f"Loading reset file: {filepath}")
        with open(filepath, 'rb') as infile:
            reset_file = pickle.load(infile)
        return reset_file
    else:
        print(f"No reset file found in {filepath}")
    return None


def read_data():
    """
    Read feature data.

    :return: list[np.ndarray], list[tuple(agent_idx, round_idx, traj_idx, clone_idx, n_frames)].
        List of featurized trajectories and indices to map from trajectory to file.
    """
    print("Reading feature data.")
    agent_dirs = natsorted(glob('agent_*'))
    n_agents = len(agent_dirs)
    print(f"Detected {n_agents} agents.")
    round_dirs = natsorted(glob(os.path.join(agent_dirs[0], 'round_*')))
    n_rounds = len(round_dirs)
    print(f"Detected {n_rounds} previous rounds.")
    trajectories = []
    frame_indices_map = []  # Keep map to find each frame later
    for a_idx, a_dir in enumerate(agent_dirs):
        round_dirs = natsorted(glob(os.path.join(a_dir, 'round_*')))
        assert (len(round_dirs) == n_rounds)  # Ensures all agents ran the same number of rounds
        for r_idx, r_dir in enumerate(round_dirs):
            traj_dirs = natsorted(glob(os.path.join(r_dir, 'traj_*')))
            n_trajs = len(traj_dirs)
            for t_idx, t_dir in enumerate(traj_dirs):
                feature_files = natsorted(glob(os.path.join(t_dir, '*.npy')))
                n_files = len(feature_files)
                for f_idx, f_file in enumerate(feature_files):
                    feats = np.load(f_file)
                    n_frames = feats.shape[0]
                    trajectories.append(feats)
                    frame_indices_map.append((a_idx + 1, r_idx + 1, t_idx + 1, f_idx + 1, n_frames))
    return trajectories, frame_indices_map


def concatenate_data(trajectories):
    """
    Concatenate trajectories for clustering and processing.

    :param trajectories: list[np.ndarray].
        Array of featurized trajectories.
    :return: np.ndarray of shape (n_frames, n_features).
        Concatenated trajectories.
    """
    return np.concatenate(trajectories, axis=0)


def cluster_data(clusters, concat_data):
    """
    Cluster the data. The clustering algorithm can be replaced here.

    :param clusters: int.
        Number of clusters.
    :param concat_data: list[np.ndarray].
        Array of featurized trajectories.
    :return: clustering object.
    """
    return KMeans(n_clusters=clusters, init='k-means++').fit(concat_data)


def define_states_aux(central_frames_indices, frame_stride, frames_indices_map, format_traj):
    """
    Auxiliary function for define_states().
    """
    frame_indices = np.asarray([f[4] for f in frames_indices_map])
    frame_index_scan = np.add.accumulate(frame_indices)
    file_indices = np.digitize(central_frames_indices, frame_index_scan)  # From frame index to file index
    states = []
    for i, f_idx in enumerate(file_indices):
        agent_idx, round_idx, traj_idx, clone_idx, _ = frames_indices_map[f_idx]
        filepath = natsorted(glob(os.path.join(f"agent_{agent_idx}",
                                               f"round_{round_idx}",
                                               f"traj_{traj_idx}",
                                               f"*.{format_traj}")))[clone_idx - 1]
        offset = 0 if (f_idx == 0) else frame_index_scan[f_idx - 1]
        frame_idx = (central_frames_indices[i] - offset) * frame_stride
        states.append((filepath, frame_idx))

    return states


def define_states(cluster_obj, n_candidates, concat_data, frame_stride, frames_indices_map, format_traj):
    """
    Define the states to be considered for simulation restarting.

    :param cluster_obj: clustering object.
        Object returned by sklearn.cluster.KMeans.
    :param n_candidates: int.
        Number of least counts candidates.
    :param concat_data: np.ndarray.
        Concatenated frames from trajectories.
    :param frame_stride: int.
        Stride used to featurize trajectory files.
    :param frames_indices_map: list[tuple(int)].
        Indices needed to map a frame index to a given file.
    :param format_traj: str.
        Trajectory file format.
    :return: list of states as tuple (filepath, frame_idx), featurized states, and cluster indices of candidate states.
    """
    counts = Counter(cluster_obj.labels_)
    least_counts = np.asarray(counts.most_common()[::-1][:n_candidates])[:, 0]
    least_counts_centers = cluster_obj.cluster_centers_[least_counts]
    central_frames_indices, _ = pairwise_distances_argmin_min(least_counts_centers, concat_data)
    states = define_states_aux(central_frames_indices, frame_stride, frames_indices_map, format_traj)
    states_feat = concat_data[central_frames_indices]
    return states, states_feat, least_counts


def compute_stakes_aux(stakes, stakes_method, stakes_k):
    """
    Auxiliary function for compute_stakes(). This function is only relevant when using a stakes_method other than
    'percentage'.
    """
    temp = stakes.copy()

    if stakes_method == "equal":
        for i in range(temp.shape[1]):
            temp[:, i][np.where(stakes[:, i] != 0)] = 1 / np.count_nonzero(stakes[:, i])

    elif stakes_method == "logistic":
        k = stakes_k
        x0 = 0.5

        def logistic_fun(x):
            return 1 / (1 + np.exp(-k * (x - x0)))

        for i in range(temp.shape[1]):
            temp[:, i][np.where(stakes[:, i] != 0)] = logistic_fun(stakes[:, i][np.where(stakes[:, i] != 0)])
            # temp[:, i][np.where(stakes[:, i] < 1e-18)] = 0  # Evaluate to zero at x < 1e-18
            temp[:, i][np.where(stakes[:, i] != 0)] /= temp[:, i][np.where(stakes[:, i] != 0)].sum()

    return temp


def compute_stakes(n_agents, n_candidates, trajectories, frames_indices_map, cluster_obj, least_counts_idx,
                   stakes_method, stakes_k):
    """
    Compute stakes for all agents.

    :param n_agents: int.
        Number of agents.
    :param n_candidates: int.
        Number of candidate states.
    :param trajectories: list[np.ndarray]
        List of trajectories.
    :param frames_indices_map: list[tuple(int)].
        Indices needed to map a frame index to a given file.
    :param cluster_obj: clustering object.
        Object returned by sklearn.cluster.KMeans
    :param least_counts_idx: list[int]
        Indices of least counts candidates.
    :param stakes_method: str.
        Method to compute stakes.
    :param stakes_k: float.
        Kappa parameter when using stakes_method = 'logistic'.
    :return: np.ndarray of shape (n_agents, n_candidates).
        Stakes array.
    """
    num_frames = np.zeros((n_agents, n_candidates))
    for traj, f_idx_map in zip(trajectories, frames_indices_map):
        agent_idx = f_idx_map[0] - 1
        assert (traj.shape[0] == f_idx_map[4])
        traj_labels = cluster_obj.predict(traj)
        for clust_idx in least_counts_idx:
            num_frames[agent_idx, clust_idx] += np.count_nonzero(traj_labels == clust_idx)
    stakes = normalize(num_frames, norm="l1", axis=0)

    return compute_stakes_aux(stakes, stakes_method, stakes_k)


def split_agent_data(trajectories, frames_indices_map, stakes, agent_idx):
    """
    Split data according to which agent it belongs to. This is necessary to compute the reward from each agent.

    :param trajectories: list[np.ndarray]
        List of trajectories.
    :param frames_indices_map: list[tuple(int)].
        Indices needed to map a frame index to a given file.
    :param stakes: np.ndarray of shape (n_agents, n_candidates).
        Stakes array.
    :param agent_idx: int.
        Index of the corresponding agent (1-indexed to follow user numbering).
    :return: np.ndarray, np.ndarray.
        Data belonging to agent agent_idx.
        Stakes of agent agent_idx.
    """
    data_agent = []
    for traj, f_idx_map in zip(trajectories, frames_indices_map):
        agent = f_idx_map[0]
        if agent == agent_idx:
            data_agent.append(traj)
    data_agent = concatenate_data(data_agent)
    stakes_agent = stakes[agent_idx - 1]

    return data_agent, stakes_agent


def compute_scores(means, stdev, stakes_agent, states_feat, cv_weights_agent):
    """
    Compute rewards for a given agent.

    :param means: np.ndarray.
        Vector of means for data observed by agent.
    :param stdev: np.ndarray.
        Vector of std deviations for data observed by agent.
    :param stakes_agent: np.ndarray.
        Stakes for the given agent.
    :param states_feat: np.ndarray.
        Featurized states.
    :param cv_weights_agent: np.ndarray.
        Collective variable weights for the agent.
    :return: np.ndarray.
        Reward values of the agent.
    """
    epsilon = sys.float_info.epsilon
    dist = np.abs(states_feat - means)
    distances = (cv_weights_agent * dist / (stdev + epsilon)).sum(axis=1)

    return stakes_agent * distances


def set_weights(cv_weights, delta, means, stdev, stakes_agent, states_feat):
    """
    Find new CV weights for a given agent.

    :param cv_weights: np.ndarray.
        Previous weights.
    :param delta: float.
        Delta parameter (same for all agents).
    :param means: np.ndarray.
        Vector of means for data observed by agent.
    :param stdev: np.ndarray.
        Vector of std deviations for data observed by agent.
    :param stakes_agent: np.ndarray
        Stakes for the given agent.
    :param states_feat: np.ndarray.
        Featurized states.
    :return: np.ndarray.
        New weights.
    """
    # Define constraints
    weights_prev = cv_weights

    constraints = [
        # Inequality constraints (fun(x, *args) >= 0)
        # This constraint makes the weights change by delta (at most)
        {
            'type': 'ineq',
            'fun': lambda weights, weights_prev, delta: delta - np.abs((weights_prev - weights)),
            'jac': lambda weights, weights_prev, delta: np.diagflat(np.sign(weights_prev - weights)),
            'args': (weights_prev, delta),
        },
        # This constraint makes the weights be always positive
        {
            'type': 'ineq',
            'fun': lambda weights: weights,
            'jac': lambda weights: np.eye(weights.shape[0]),
        },
        # Equality constraints (fun(x, *args) = 0)
        # This constraint makes sure the weights add up to one
        {
            'type': 'eq',
            'fun': lambda weights: weights.sum() - 1,
            'jac': lambda weights: np.ones(weights.shape[0]),
        }]

    def minimize_helper(x):
        return -compute_scores(means, stdev, stakes_agent, states_feat, x).sum()

    results = minimize(minimize_helper, weights_prev, method='SLSQP', constraints=constraints)

    return results.x


def select_states(scores, stakes, interaction_regime, n_select):
    """
    Aggregate scores and choose states with the highest rewards.

    :param scores: np.ndarray.
        Scores assigned by all agents.
    :param stakes: np.ndarray.
        Stakes for all agents.
    :param interaction_regime: str.
        Method to aggregate scores.
    :param n_select: int.
        Number of states to select for next round.
    :return: np.ndarray, np.ndarray.
        Selected state indices.
        Agents that are assigned the new trajectories.
    """
    # Aggregate scores
    if interaction_regime == "collaborative":
        aggregated_scores = scores.sum(axis=0)
    elif interaction_regime == "noncollaborative":
        aggregated_scores = scores.max(axis=0)
    elif interaction_regime == "competitive":
        aggregated_scores = 2 * scores.max(axis=0) - scores.sum(axis=0)

    # Select maxima
    state_indices = np.argsort(aggregated_scores)[-n_select:][::-1]
    executors = np.argmax(stakes, axis=0)[state_indices]

    return state_indices, executors


def save_files(selected_states, executors, format_coor, topology, n_round, n_agents):
    """
    Create input files for next round of simulations.

    :param selected_states: list[tuple()].
        Indices required to map selected states to source files.
    :param executors: list[int].
        Agents that will run each new trajectory. Note executors are 0-indexed, unlike agents which are 1-indexed.
    :param format_coor: str.
        Format for new input files.
    :param topology: str.
        Path to topology file.
    :param n_round: int.
        Current simulation round.
    :param n_agents: int.
        Number of agents.
    :return: None.
    """
    traj_counter = np.ones((n_agents), dtype=int)
    csv_file = 'Input File, Source, Source Frame Idx\n'

    # Create new round directories
    for a in range(n_agents):
        os.makedirs(os.path.join(f'agent_{a + 1}', f'round_{n_round + 1}'), exist_ok=True)

    for state, executor in zip(selected_states, executors):
        filepath, frame_idx = state
        coordinates = md.load_frame(filepath, frame_idx, top=topology)
        outpath = os.path.join(f'agent_{executor + 1}',
                               f'round_{n_round + 1}',
                               f'traj_{traj_counter[executor]}'
                               )
        os.makedirs(outpath, exist_ok=True)
        outfile = os.path.join(outpath, f'input.{format_coor}')
        coordinates.save(outfile)
        traj_counter[executor] += 1
        csv_file += f'{outpath}, {filepath}, {frame_idx}\n'

    with open(f'input_states_for_round_{n_round + 1}.csv', 'w') as outfile:
        outfile.write(csv_file)


def write_reset_file(n_round, topology, format_traj, format_coor, frame_stride, delta, weights, n_candidates,
                     n_output, clusters, stakes_method, stakes_k, regime):
    """
    Save reset file.

    :return: None.
    """
    reset = dict(
        n_round=n_round,
        topology=topology,
        format_traj=format_traj,
        format_coor=format_coor,
        frame_stride=frame_stride,
        delta=delta,
        weights=weights,
        n_candidates=n_candidates,
        n_output=n_output,
        clusters=clusters,
        stakes_method=stakes_method,
        stakes_k=stakes_k,
        regime=regime,
    )

    with open(f'restart_file_round_{n_round}.pkl', 'wb') as outfile:
        pickle.dump(reset, outfile)


def main():
    parser = set_parser()
    reset_file = read_reset_file(parser.reset)

    if reset_file:
        # Check that all settings can be found in reset file
        topology = reset_file['topology']
        format_traj = reset_file['format_traj']
        format_coor = parser['format_coor'] if parser['format_coor'] else reset_file['format_coor']
        frame_stride = reset_file['frame_stride']
        delta = reset_file['delta']
        weights = reset_file['weights']
        n_candidates = parser['n_candidates'] if parser['n_candidates'] else reset_file['n_candidates']
        n_output = parser['n_output'] if parser['n_output'] else reset_file['n_output']
        clusters = parser['clusters'] if parser['clusters'] else reset_file['clusters']
        stakes_method = reset_file['stakes_method']
        stakes_k = reset_file['stakes_k']
        regime = reset_file['regime']
        n_round = reset_file['n_round']
    else:
        # Take user inputs
        topology = parser['topology']
        format_traj = parser['format_traj']
        format_coor = parser['format_coor']
        frame_stride = parser['frame_stride']
        delta = parser['delta']
        weights = parser['weights']
        n_candidates = parser['n_candidates']
        n_output = parser['n_output']
        clusters = parser['clusters']
        stakes_method = parser['stakes_method']
        stakes_k = parser['stakes_k']
        regime = parser['regime']
        n_round = 1  # Forced

        # Some assertions to check user's input
        assert (os.path.exists(topology))
        assert (0 < delta < 1)
        assert (np.allclose(sum(weights), 1))
        assert (n_output <= n_candidates <= clusters)
        if stakes_method == 'logistic':
            assert (stakes_k is not None)

    trajectories, frames_indices_map = read_data()
    n_agents = frames_indices_map[-1][0]

    # Check that all feature files have the same number of features
    n_features = trajectories[0].shape[1]
    for traj in trajectories:
        assert (traj.shape[1] == n_features)

    concat_data = concatenate_data(trajectories)
    cluster_obj = cluster_data(clusters, concat_data)
    states, states_feat, least_counts_idx = define_states(cluster_obj, n_candidates, concat_data, frame_stride,
                                                          frames_indices_map, format_traj)
    stakes = compute_stakes(n_agents, n_candidates, trajectories, frames_indices_map, cluster_obj, least_counts_idx,
                            stakes_method, stakes_k)

    # Set new weights and compute scores for each agent
    weights = np.asarray(weights)
    if weights.shape == (n_features,):
        prev_weights = np.asarray([weights for _ in range(n_agents)])
        assert (prev_weights.shape == (n_features, n_agents))
    elif weights.shape == (n_features, n_agents):
        prev_weights = weights
    new_weights = np.empty(prev_weights.shape)
    scores = np.empty((n_agents, n_candidates))
    for a_idx in range(1, n_agents + 1):
        data_agent, stakes_agent = split_agent_data(trajectories, frames_indices_map, stakes, a_idx)
        means = data_agent.mean(axis=0)
        stdev = data_agent.std(axis=0)
        new_weights[a_idx - 1] = set_weights(prev_weights, delta, means, stdev, stakes_agent, states_feat)
        scores[a_idx - 1] = compute_scores(means, stdev, stakes_agent, states_feat, new_weights[a_idx - 1])

    # Find states and save new input files
    state_indices, executors = select_states(scores, stakes, regime, n_output)
    save_files(states[state_indices], executors, format_coor, topology, n_round, n_agents)
    # Write new reset file
    write_reset_file(n_round + 1, topology, format_traj, format_coor, frame_stride, delta, new_weights, n_candidates,
                     n_output, clusters, stakes_method, stakes_k, regime)


if __name__ == '__main__':
    main()
