# Running simulations
To run simulations (from `Langevin dynamics/<name-of-method>/`):

```
conda activate mareap
python main.py <int>
```

The argument passed to `main.py` is just used to distinguish runs when running trials in parallel.

Parameters used in the simulations from the paper are provided in `settings.py` files (LC, REAP, MA REAP) or within the `main.py` files for TSLC and MA TSLC.

## Trying other potentials
If you wish to try other toy potentials, you must define two new things:
- `potential`: a string with the equation defining the potential (must be admitted by OpenMM's `CustomExternalForce`). Example: `"2*(x-1)^2*(x+1)^2 + y^2"`.
- `potential_func`: a function that returns the value of the potential at the given particle position. Example: `f = lambda x, y: 2*(x-1)**2*(x+1)**2 + y**2`. This is used for plotting the potential and calculating the covered area.

Other parameters you will probably need to change:
- `threshold`: higher energy values will not contribute towards the "discoverable" area
- `xlim/ylim`: limits for plotting potential/computing covered area