# Shared variables
CONDA_ROOT=/iris/u/abhijnya/miniconda3
CROSS_PREFERENCES_DIR="/iris/u/abhijnya/FPL/test_cross_pref_path"
PREFERENCES_DIR="/iris/u/abhijnya/FPL/test_pref_path"
OUTPUT_ROOT=/iris/u/abhijnya/FPL/real_world/infer_output
OPENPI_DIR="/iris/u/abhijnya/FPL/real_world/openpi"
OPENPI_PY="$OPENPI_DIR/.venv/bin/python"

export WANDB_API_KEY="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"
export HF_LEROBOT_HOME=/iris/u/abhijnya/data
export WANDB_SERVICE_WAIT=120
export WANDB_START_METHOD=thread
export HOME=/iris/u/abhijnya
source /iris/u/abhijnya/miniconda3/etc/profile.d/conda.sh
source $HOME/.local/bin/env 