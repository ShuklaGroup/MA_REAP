# Multi-agent reinforcement learning-based adaptive sampling
If you use this code, make sure to cite:

```BibTeX
@article {Kleiman2022.05.31.494208,
	author = {Kleiman, Diego E and Shukla, Diwakar},
	title = {Multi-Agent Reinforcement Learning-based Adaptive Sampling for Conformational Sampling of Proteins},
	elocation-id = {2022.05.31.494208},
	year = {2022},
	doi = {10.1101/2022.05.31.494208},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/05/31/2022.05.31.494208},
	eprint = {https://www.biorxiv.org/content/early/2022/05/31/2022.05.31.494208.full.pdf},
	journal = {bioRxiv}
}
```
  
## Environment setup
The easiest way to install the required dependencies is through [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):

```
conda env create -f mareap.yml
```

## Repository organization
The codes are sorted by simulation type (Langevin, KMC, all-atom MD) and adaptive sampling techinque implemented.

Follow the instructions on each folder's README to run the simulations. Note that the code provided here is for 
experimental purposes only and has not been optimized (trajectories are run serially).
