#!/bin/bash
set -e

if [ -d /iris/u/marcelto/miniconda3 ]; then
    CONDA_ROOT=/iris/u/marcelto/miniconda3
elif [ -d /hai/scratch/marcelto/miniconda3 ]; then
    CONDA_ROOT=/hai/scratch/marcelto/miniconda3
else
    echo "ERROR: could not find miniconda on /iris or /hai" >&2
    exit 1
fi
eval "$(${CONDA_ROOT}/bin/conda shell.bash hook)"

export MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

conda activate robodiffrew2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/fetch_results.py" "$@"
