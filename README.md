# FPL — Freeform Preference Learning
This Repository contains the code for Reward Learning and Simulation

[![arXiv](https://img.shields.io/badge/arXiv-2602.23359-b31b1b.svg)](https://github.com/freeform-preference-learning)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://github.com/freeform-preference-learning)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/freeform-preference-learning)


## Simulation Experiments

### Installation
Create the environment using the following commands:
```bash
cd diffusion_policy
conda env create -f conda_environment.yaml
conda activate robodiff

```
Export your WandB API Key
```bash
export WANDB_API_KEY=your_key_here  # get it from https://wandb.ai/settings
```

Download the dataset from the [robomimic website](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html):
```bash
python -m robomimic.scripts.download_datasets --tasks square --dataset_types mh --hdf5_types low_dim --download_dir data/robomimic/datasets
```


### Running the pipeline

#### Bimodal Square task:
```bash
./scripts/slow_fast/launch_rhp.sh
```

#### Object Rearrangement Task:
```bash
./scripts/object_rearrangement/launch_rhp.sh
```
