# Running simulations
To run simulations (from `All-atom MD/MA REAP/`):

```
conda activate mareap
python main.py
```

Simulation parameters in `main.py` can be edited as necessary.

## Simulating other systems
We provide an alanine dipeptide topology file and starting conformations in `example_data/`.

If you wish to simulate a different system, you need to change the following things:

`main.py`:
- Pass your topology filename in the `kwargs`
- Pass appropriate starting structures through `initial_structures` (usually, `n_agents` should be set to the number of starting conformations you provide but you may want to experiment with this)

`MA_REAP.py`:
- Provide a function that returns your [OpenMM](http://docs.openmm.org/latest/api-python/) system, topology, integrator, and initial coordinates (replaces `alanine_dipeptide_system()`)
- Use the function form the first point in `spawn_trajectories()`
- Provide a function that projects your system into the desired collective variables (replaces `project_dihedral_alanine_dipeptide()`)
- Use the function from the previous point in `collect_initial_data()` and `run_trial()`
- (Optional) Change plotting parameters in `run_trial()` according to your system
