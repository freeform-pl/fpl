#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=single_pref_slow_fast
#SBATCH --nodelist=iris9,iris10
#SBATCH --output slurm/%j.out

# Single-pref baseline for slow/fast experiment (1 composite reward dim)
export PIPELINE_DIR="pipeline_output_slow_fast_single_pref"
export WANDB_PROJECT="slow_fast_single_pref"
export NUM_REWARD_DIMS=1
export REWARD_AXES="composite"
export RESUME_FROM_PHASE=0

# Per-axis eval z-score conditioning (length must match NUM_REWARD_DIMS=1)
export EVAL_Z_POSITIVE="[1.0]"
# export EVAL_Z_NEGATIVE="[-1.0]"

bash scripts/run_pipeline_slow_fast.sh
