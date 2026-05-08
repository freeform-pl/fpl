#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=demo_success_slow_fast
#SBATCH --nodelist=iris4,iris5,iris6,iris7,iris8
#SBATCH --output slurm/%j.out

# Demo-success baseline for slow/fast experiment (no reward model, uses demo labels directly)
export PIPELINE_DIR="pipeline_output_slow_fast_demo_success"
export WANDB_PROJECT="slow_fast_demo_success"
export COND_CONFIG="train_demo_success_flow_transformer_lowdim_workspace.yaml"
export SKIP_REWARD_MODEL=true
export IS_CONDITIONED_EVAL=false
export RESUME_FROM_PHASE=1

bash scripts/run_pipeline_slow_fast.sh
