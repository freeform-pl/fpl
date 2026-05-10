#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=awr_slow_fast
#SBATCH --nodelist=iris4,iris5,iris6,iris7,iris8
#SBATCH --output slurm/%j.out

# AWR baseline for slow/fast experiment (3 reward dims, advantage-weighted regression)
export PIPELINE_DIR="pipeline_output_slow_fast_awr"
export WANDB_PROJECT="slow_fast_awr"
export COND_CONFIG="train_awr_flow_transformer_lowdim_workspace.yaml"
export IS_CONDITIONED_EVAL=false
export RESUME_FROM_PHASE=0

bash scripts/run_pipeline_slow_fast.sh
