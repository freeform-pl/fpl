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

# python main.py --stride 8 --seq_len 14 --img_size 128 --batch_size 32 --epochs 200 --embed_dim 128 --vis_interval 1 --save_interval 10 --preferences_dir preferences_2026_04_11,preferences_2026_04_14
python main.py --stride 10 --seq_len 20 --img_size 128 --batch_size 64 --epochs 4000 --embed_dim 128 --vis_interval 50 --log_interval 1 --save_interval 50 --preferences_dir preferences_2026_04_11,preferences_2026_04_14 --model flow --preload
# python main.py --stride 8 --seq_len 14 --img_size 128 --batch_size 8 --epochs 200 --embed_dim 128 --vis_interval 1 --save_interval 10 --preferences_dir preferences_2026_04_11,preferences_2026_04_14