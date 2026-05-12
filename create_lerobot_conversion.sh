#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=16 # Request 8 CPUs for this task
#SBATCH --mem=128G # Request 8GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name=infer # Name the job (for easier monitoring)
#SBATCH --nodelist=iris1,iris2,iris3,iris4# Don't run on iris1
#SBATCH --output slurm/%j.out # MAKE SURE slurm/ ALREADY EXISTS, OR ELSE YOUR JOB WILL FAIL SILENTLY!

# Now your Python or general experiment/job runner code
cd /iris/u/marcelto/reward_learning
export HOME=/iris/u/marcelto
eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate lerobot

python convert_custom_droid_to_lerobot.py \
    --args.scores_dir /iris/u/marcelto/reward_learning/infer_output/setup_table_multi/2026-05-07_21-50-42_transformer_j15364912_2100 \
    --args.repo_name marcelto/setup_table_multi_standardized_1dp \
    --args.task_prompt "set up the table" \
    --args.score_type standardized \
    --args.decimal_places 1 


python convert_custom_droid_to_lerobot.py \
    --args.scores-dir /iris/u/marcelto/reward_learning/infer_output/fold_pants_multi/2026-05-07_22-42-32_transformer_j15366663_1400 \
    --args.repo-name marcelto/fold_pants_multi_standardized_1dp \
    --args.task-prompt "fold the shorts" \
    --args.score-type standardized \
    --args.decimal-places 1