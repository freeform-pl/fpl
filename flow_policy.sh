#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris # Run on IRIS nodes
#SBATCH --time=24:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=8 # Request 8 CPUs for this task
#SBATCH --mem=32G # Request 8GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name=reward_learning # Name the job (for easier monitoring)
#SBATCH --nodelist=iris6,iris5,iris7# Don't run on iris1
#SBATCH --output slurm/%j.out # MAKE SURE slurm/ ALREADY EXISTS, OR ELSE YOUR JOB WILL FAIL SILENTLY!


# conda env create -f conda_environment.yaml

# pip install \
#   "ray[default,tune]==2.2.0" \
#   free-mujoco-py==2.1.6 \
#   pygame==2.1.2 \
#   pybullet-svl==3.1.6.4 \
#   "robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz" \
#   robomimic==0.2.0 \
#   pytorchvideo==0.1.5 \
#   imagecodecs==2022.9.26 \
#   "r3m @ https://github.com/facebookresearch/r3m/archive/b2334e726887fa0206962d7984c69c5fb09cceab.tar.gz" \
#   dm-control==1.0.9                          
eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"


export MUJOCO_PATH=~/.mujoco/mujoco210                                                                                             
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin                                                                    

conda activate robodiffrew2                                                       

cd diffusion_policy

python train.py --config-name=train_flow_transformer_lowdim_workspace.yaml task=square_lowdim task.dataset_type=mh