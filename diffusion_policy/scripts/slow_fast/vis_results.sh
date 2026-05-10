#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=success_right_peg_slow_fast
#SBATCH --nodelist=iris4,iris5,iris6,iris7,iris8
#SBATCH --output slurm/%j.out


eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate robodiffrew2

python scripts/plot_wandb_comparison.py \
  --runs rhp=memory_rl/slow_fast_rhp/pb05ve1b \
         demo_only=memory_rl/slow_fast_demo_only/9b2py8np \
         awr=memory_rl/slow_fast_awr/txbjh9nj \
         demo_success=memory_rl/slow_fast_demo_success/pc395vnn \
         single_pref=memory_rl/slow_fast_single_pref/bl23g53m \
         success_right_peg=memory_rl/slow_fast_success_right_peg/bcta32uq \
  --output_dir plots/slow_fast_comparison
