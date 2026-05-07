#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=rhp
#SBATCH --nodelist=iris4,iris5,iris6,iris7
#SBATCH --output slurm/%j.out

export BASELINE=rhp
export RESUME_FROM_PHASE=3

# Per-axis eval z-score conditioning (length must match num_reward_dims=3: success, speed, smoothness)
export EVAL_Z_POSITIVE="[0.5,1,1.5]"
# export EVAL_Z_NEGATIVE="[-1.5,-1.5,-1.5]"

bash scripts/run_pipeline.sh
