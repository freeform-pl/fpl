#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=singlepref
#SBATCH --nodelist=iris4,iris5,iris6,iris7
#SBATCH --output slurm/%j.out

export BASELINE=single_pref
export RESUME_FROM_PHASE=3

# Per-axis eval z-score conditioning (length must match num_reward_dims=1: composite)
# export EVAL_Z_POSITIVE="[1.5]"
# export EVAL_Z_NEGATIVE="[-1.5]"

bash scripts/run_pipeline.sh
