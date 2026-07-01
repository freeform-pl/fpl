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
./scripts/slow_fast/launch_fpl.sh
```

#### Object Rearrangement Task:
```bash
./scripts/object_rearrangement/launch_fpl.sh
```


## Real World Experiments
This takes you through training the Qwen reward model on the collected preference pairs, converting them into LeRobot format and then filetuning pi05 with this data

We assume that the preferences pairs and optional cross preferences have been collected. For more information on this refer [this repo](https://github.com/freeform-pl/fpl_real)

### Installation
We will require 2 environments for this.

1. Environment for launching Qwen training and inference scripts
Create the environment using the following commands:
```bash
cd real_world
conda env create -f conda_environment_real.yaml
```

2. Finetuning Pi05:
Follow [this repo](https://github.com/Physical-Intelligence/openpi) to download the code and set up the pi05 environment.

### Running the pipeline

#### Training the reward model
```bash
./scripts/train_qwen.sh
```


#### Inferring from the reward model and finetuningPi05
```bash
./scripts/infer_then_train_pi05.sh
```


## TO DO
- [ ] Remove slurm related code from sh files
- [ ] Remove absolute paths, add relative path for sourcing config.sh file in train_qwen.sh and infer_then_train_pi05.sh
- [ ] Add readme for mentioning the variables that need to be set in config.sh
- [ ] Replace all paths with `<your xyz path>`
- [ ] Enhance the ReadMe to add directory structure and more information on running it seamlessly.


