# Multi-agent reinforcement learning-based adaptive sampling
If you use this code, make sure to cite:

```BibTeX
@article{kleiman2022multiagent,
  title={Multiagent Reinforcement Learning-Based Adaptive Sampling for Conformational Dynamics of Proteins},
  author={Kleiman, Diego E and Shukla, Diwakar},
  journal={Journal of Chemical Theory and Computation},
  year={2022},
  publisher={ACS Publications}
}
```

## NEW (Jan 2023): CLI to run MA REAP simulations with any MD engine

A Python script (`mareap_sim.py`) with a command line interface is available to simplify the use of MA REAP with arbitrary MD engines (with the condition that the trajectory/coordinate files can be understood by [mdtraj](https://mdtraj.org/1.9.4/load_functions.html)). This script does not run any trajectories, but it generates the input files required to perform a round of simulations. The expected use case is that the user will call this script after each round of simulations to obtain the input files for the following round of adaptive sampling.  

To install dependencies, you can use the conda environment file `mareap_sim.yml` (it contains fewer dependencies than `mareap.yml`):

```
conda env create -f mareap_sim.yml
```

The script will search for trajectory files in its working directory, which must follow the following structure:

```
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
```

New directories `round_{M+1}` are created under each agent with the corresponding input coordinate files stored in `traj_{#}` subdirectories. The file `input_states_for_round_{round #}.csv` specifies the trajectory and frame that was read to obtain the input coordinates.
The user is responsible for extracting the relevant features from the trajectories and store them in the `.npy` format. For instructions on trajectory featurization, [this](https://mdtraj.org/1.9.4/analysis.html) and [this](https://userguide.mdanalysis.org/stable/examples/quickstart.html) might be useful. Multiple clones (i.e., trajectories initialized from the exact same conditions) can be stored under each `traj_{#}` folder, but to make sure that the features are associated to the correct trajectory, it is recommended that the names of the files match. 

The flags that can be set by the user are the following:

```
-r RESET, --reset RESET
                        Path to MAREAP reset file (.pkl). This file is created when running the script for the first time. Some settings are ignored if passing a reset file.
  -t TOPOLOGY, --topology TOPOLOGY
                        Path to the topology file. (Ignored if passing reset file.)
  -ft FORMAT_TRAJ, --format_traj FORMAT_TRAJ
                        Format suffix for trajectory files. (Ignored if passing reset file.)
  -fc FORMAT_COOR, --format_coor FORMAT_COOR
                        Format suffix for new input files.
  -fs FRAME_STRIDE, --frame_stride FRAME_STRIDE
                        Frame stride used when featurizing trajectories. (Ignored if passing reset file.)
  -d DELTA, --delta DELTA
                        Delta parameter for MA REAP. Default: 0.05. (Ignored if passing reset file.)
  -w WEIGHTS [WEIGHTS ...], --weights WEIGHTS [WEIGHTS ...]
                        Initial feature weights for MA REAP (list of floats must add to 1). (Ignored if passing reset file.)
  -n N_CANDIDATES, --n_candidates N_CANDIDATES
                        Number of least count candidates to consider.
  -o N_OUTPUT, --n_output N_OUTPUT
                        Number of trajectories for next round of simulations.
  -c CLUSTERS, --clusters CLUSTERS
                        Number of clusters to use (KMeans).
  -s {percentage,equal,logistic}, --stakes_method {percentage,equal,logistic}
                        Method to calculate stakes. Default: percentage. (Ignored if passing reset file.)
  -sk STAKES_K, --stakes_k STAKES_K
                        k parameter when using logistic stakes. Only needs to be set when using -s logistic. (Ignored if passing reset file.)
  -reg {collaborative,noncollaborative,competitive}, --regime {collaborative,noncollaborative,competitive}
                        Regime used to combine rewards from different agents. Default: collaborative. (Ignored if passing reset file.)
```

Once a reset file is available, its path must be passed to the `-r` flag and all settings will be set automatically. The user can override flags `-fc`, `-n`, `-o`, and `-c` to accomodate for certain changes.
  
## Environment setup
The easiest way to install the required dependencies is through [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):

```
conda env create -f mareap.yml
```

## Repository organization
The codes are sorted by simulation type (Langevin, KMC, all-atom MD) and adaptive sampling techinque implemented.

Follow the instructions on each folder's README to run the simulations. Note that the code provided here is for 
experimental purposes only and has not been optimized (trajectories are run serially).
