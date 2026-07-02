# FPL — Freeform Preference Learning
This Repository contains the code for Reward Learning and Simulation

[![arXiv](https://img.shields.io/badge/arXiv-2602.23359-b31b1b.svg)](https://github.com/freeform-preference-learning)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://freeform-pl.github.io/fpl.website/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/freeform-pl)


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
This guide walks through training the Qwen reward model on collected preference pairs, converting the data to LeRobot format, and finetuning Pi05.


Before proceeding, preference pairs (and optionally cross preferences) must already be collected. See [this repo](https://github.com/freeform-pl/fpl_real) for details on how to do this.

### Installation
We will require 2 environments for this:

1. Environment for launching Qwen training and inference scripts
Create the environment using the following commands:
```bash
cd real_world
conda env create -f conda_environment_real.yaml
```

2. Environment for Finetuning Pi05
Create the environments using the following commands:
```bash
cd openpi
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
For more information and troubleshooting follow the instructions on [this repo](https://github.com/Physical-Intelligence/openpi).

### Running the pipeline

#### Step 1: Training the reward model
First, open `./scripts/config.sh` and update the following variables:
```bash
TASK=""                       # Task to train on — see tasks.py for the full list
TASK_PROMPT=""                # Natural language prompt describing the task
DEMOS_DIR=""                  # Comma-separated paths to teh demo directories
PREFERENCES_DIR=""            # Comma-separated paths to preferences directories
CROSS_PREFERENCES_DIR=""      # Comma-separated paths to cross preferences directories
export WANDB_API_KEY=your_key_here  # get it from https://wandb.ai/settings
```
Then, train the reward model by running to following:
```bash
./scripts/train_qwen.sh
```

We will use the checkpoint so obtained for the next step.


#### Step 2: Inferring from the reward model and finetuning Pi05
Now that we have our reward model trained, we will use it to obtain scores for all the data to then finetunie Pi05

Open `./scripts/infer_then_train_pi05.sh` and update the checkpoint path to the path obtained after training the reward model:
```bash
CKPT=exp/<your folder>/checkpoints/final.pt     # Latest training checkpoint
```

Then run the following script:
```bash
./scripts/infer_then_train_pi05.sh
```
This script is responsible for infering scores from the reward model, converting them into the desired format, and finetuning Pi05 using that data.


## Acknowledgements
The development of this project, largely benefitted and directly used the following repositories:
- [openpi](https://github.com/Physical-Intelligence/openpi/tree/main) for finetuning the real world policies and open source weights of Pi05.
- [robomimic](https://robomimic.github.io/) for the simulation structure.
- [diffusion policy](https://github.com/real-stanford/diffusion_policy) for the code and simulation structure.
- [qwen](https://huggingface.co/Qwen/Qwen3.5-4B) for their VLM open source weights.

We are very grateful for the great work done with open sourcing the mentioned projects.
