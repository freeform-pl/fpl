#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --mem=32G # Request 8GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name=reward_learning # Name the job (for easier monitoring)
#SBATCH --nodelist=iris9,iris10,iris8,iris7,iris6# Don't run on iris1
#SBATCH --output slurm/%j.out # MAKE SURE slurm/ ALREADY EXISTS, OR ELSE YOUR JOB WILL FAIL SILENTLY!

# Now your Python or general experiment/job runner code
cd /iris/u/marcelto/reward_learning
export HOME=/iris/u/marcelto
eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
source .venv/bin/activate

python infer.py --ckpt exp/2026-04-25_15-01-08_transformer_j15246410/checkpoints/step001500.pt --preferences_dir /iris/u/am208/reward_learning_combined_success --output_dir /iris/u/marcelto/reward_learning/infer_output/2026-04-25_15-01-08_step001500 

python infer.py --ckpt exp/2026-04-22_13-38-42_transformer_scr0.2_j15219616/checkpoints/step000150.pt --preferences_dir preferences_04_11to22,preferences_2026_04_14

python infer.py --ckpt exp/2026-05-02_07-12-25_transformer_j15316054/checkpoints/step000850.pt --preferences_dir /iris/u/abhijnya/droid-robot/demos/test,/iris/u/am208/droid-robot/demos/test,/iris/u/am208/droid-robot/preferences --output_dir /iris/u/marcelto/reward_learning/infer_output/2026-05-02_07-12-25_transformer_j15316054


