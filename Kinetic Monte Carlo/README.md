# Running simulations
To run simulations (from `Kinetic Monte Carlo/<name-of-method>/`):

```
conda activate mareap
python main.py
```

Simulation parameters in `main.py` can be edited as necessary. We have provided the settings used for the systems defined in `example_data/Src` and `example_data/OsSWEET2b` (comment in/out the example you want to run).

## Simulating other systems
If you wish to simulate a different system you must provide:
- [PyEMMA](http://emma-project.org/latest/) Markov state model object or transition probability matrix (used to initialize PyEMMA MSM object) (`msm`)
- Array mapping each MSM state to CV-space vector (`state_coordinates`)
- (May be required) Array mapping state index (from largest connected component) to state index in full MSM (`mapping`)
- Input initial states (by index of full MSM). Usually, `n_agents` would be set to the number of intial states.
