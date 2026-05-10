#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=demo_only_slow_fast
#SBATCH --nodelist=iris4,iris5,iris6,iris7,iris8
#SBATCH --output slurm/%j.out

# Demo-only baseline for slow/fast experiment (just evaluate the base policy, no conditioning)
export PIPELINE_DIR="pipeline_output_slow_fast_demo_only"
export WANDB_PROJECT="slow_fast_demo_only"
export SKIP_REWARD_MODEL=true
export SKIP_POLICY_TRAINING=true
export IS_CONDITIONED_EVAL=false
export RESUME_FROM_PHASE=0

bash scripts/run_pipeline_slow_fast.sh
