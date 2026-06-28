#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=sf_awr
#SBATCH --output slurm/%j.out

# AWR baseline for slow_fast. Uses the same 2-D reward axes as RHP but the
# AWR dataset averages them into a scalar advantage weight. No reward
# conditioning at eval.
export PIPELINE_DIR="pipeline_output_slow_fast_awr"
export WANDB_PROJECT="slow_fast_awr"
export BASE_POLICY_DIR="base_policy_slow_fast"
export COND_CONFIG="train_awr_flow_transformer_lowdim_workspace.yaml"
export IS_CONDITIONED_EVAL=false
export DISCRETE_CONDITIONING=false


export SKIP_ROLLOUTS=true

export SHARED_DATA_DIR="shared_data_slow_fast"

export REWARD_AXES="speed_reward,peg_reward"
export NUM_REWARD_DIMS=2
export REWARD_EPOCHS=200
export COND_POLICY_EPOCHS=1500
# BATCH_SIZE / LEARNING_RATE intentionally left unset to use the AWR
# workspace YAML defaults (batch=256, lr=1e-4) — matches object_rearrangement
# AWR which also uses the YAML defaults.
export TRAINING_SEED=82
export EXTRA_POLICY_OVERRIDES="${EXTRA_POLICY_OVERRIDES} ++training.rollout_every=100 ++training.checkpoint_every=100"

export N_PAIRS=100

# Phase 3 trains reward model; Phase 4 trains AWR policy.
export RESUME_FROM_PHASE=4
bash scripts/run_pipeline_slow_fast.sh
