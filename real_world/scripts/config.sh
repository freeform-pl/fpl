#!/bin/bash

# ============================================================
# USER INPUTS — Change these as per your code and data paths
# ============================================================
TASK=fold_pants                             # See tasks.py for task list
TASK_PROMPT="fold the shorts"               # Add the task prompt here
DEMOS_DIR=""                                # Add the path to the demos directory here
PREFERENCES_DIR=""                          # Add the path to the preferences directory here
CROSS_PREFERENCES_DIR=""                    # Add the path to the cross preferences directory here
export WANDB_API_KEY=your_key_here          # Get it from https://wandb.ai/settings


# ============================================================
# DERIVED PATHS — No need to edit
# ============================================================
REAL_WORLD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_DIR="$REAL_WORLD_DIR/openpi"
OPENPI_PY="$OPENPI_DIR/.venv/bin/python"
OUTPUT_ROOT="$REAL_WORLD_DIR/infer_output"
export HF_LEROBOT_HOME="$REAL_WORLD_DIR/data"