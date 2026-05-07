#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=awr
#SBATCH --nodelist=iris4,iris5,iris6,iris7
#SBATCH --output slurm/%j.out

export BASELINE=awr
export RESUME_FROM_PHASE=3
bash scripts/run_pipeline.sh
