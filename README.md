# DAODEN
Implementation of a Data-Assimilation-based ODE Network (DAODEN) (https://arxiv.org/abs/2009.02296)

#### Directory Structure
The elements of the code are organized as follows:

```
daoden_datasets.py                # script to handle DAODEN datasets.
daoden_utils.py                   # DAODEN utilities.
models.py                         # DAODEN models and functions.
log_manager.py                    # logging.
daoden_main_L63.ipynb             # notebook for experiments with L63 and L63s.
daoden_main_L96.ipynb             # notebook for experiments with the L96.
datasets
├── data_generator_L63.ipynb      # notebook to generates L63 datasets.
├── data_generator_L63s.ipynb     # notebook to generates L63s datasets.
└── data_generator_L96.ipynb      # notebook to generates L96 datasets.
```

#### Requirements: 
See requirements.yml

### Datasets:

To generate the datasets, run the notebooks in `./datasets`.

### Running different models

For experiments with L63 and L63s, run `daoden_main_L63.ipynb`.
For experiments with L96, run `daoden_main_L96.ipynb`.

We also uploaded some pretrained models.

### Acknowledgement

This work was supported by Labex Cominlabs (grant SEACS), CNES (grant OSTST-MANATEE), Microsoft (AI EU Ocean awards), ANR Projects Melody and OceaniX and GENCI-IDRIS (Grant 2020-101030).

We would like to thank the author of AnDA (https://github.com/ptandeo/AnDA), SINDy (https://github.com/dynamicslab/pysindy) and Latent-ODE (https://github.com/YuliaRubanova/latent_ode) for sharing their codes.

### Contact
For any questions, please open an issue.
