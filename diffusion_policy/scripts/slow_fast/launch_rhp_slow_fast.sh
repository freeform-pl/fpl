#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=rhp_slow_fast
#SBATCH --nodelist=iris4,iris5,iris6,iris7,iris8
#SBATCH --output slurm/%j.out

# RHP baseline for slow/fast experiment (3 reward dims: speed, smoothness, peg)
export PIPELINE_DIR="pipeline_output_slow_fast_rhp"
export WANDB_PROJECT="slow_fast_rhp"
export RESUME_FROM_PHASE=4

# Per-axis eval z-score conditioning (length must match NUM_REWARD_DIMS=3: speed_reward, smoothness, peg_reward)
export EVAL_Z_POSITIVE="[1.0,1.0,1.0]"
# export EVAL_Z_NEGATIVE="[-1.0,-1.0,-1.0]"

bash scripts/run_pipeline_slow_fast.sh
