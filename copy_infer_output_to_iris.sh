#!/bin/bash
# Copy a single infer_output run from haic to iris-ws-7.
#
# Usage:
#   ./copy_infer_output_to_iris.sh <folder_name> <run_name>
#
# Example:
#   ./copy_infer_output_to_iris.sh pick_and_place_iter1_multi_qwen_v3 2026-05-21_18-26-29_qwen_open_j78841_1700
#
# Source:      /hai/scratch/marcelto/reward_learning/infer_output/<folder>/<run>
# Destination: iris-ws-7:/iris/u/marcelto/reward_learning/infer_output/<folder>/<run>
#
# Uses SSH ControlMaster so the password is only prompted ONCE.
# rsync -a is idempotent (resumes via --partial, skips already-transferred files).

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <folder_name> <run_name>"
    echo "Example: $0 pick_and_place_iter1_multi_qwen_v3 2026-05-21_18-26-29_qwen_open_j78841_1700"
    exit 1
fi

FOLDER="$1"
RUN="$2"

SRC_BASE=/hai/scratch/marcelto/reward_learning/infer_output
SRC_PATH="$SRC_BASE/$FOLDER/$RUN"

DEST_USER_HOST=marcelto@iris-ws-7.stanford.edu
DEST_BASE=/iris/u/marcelto/reward_learning/infer_output
DEST_PATH="$DEST_BASE/$FOLDER/$RUN"

if [ ! -d "$SRC_PATH" ]; then
    echo "ERROR: source run does not exist: $SRC_PATH"
    echo "Available runs in $SRC_BASE/$FOLDER:"
    ls "$SRC_BASE/$FOLDER" 2>/dev/null || echo "  (folder not found)"
    exit 1
fi

SOCK=/tmp/iris-cm-$$
cleanup() {
    ssh -S "$SOCK" -O exit "$DEST_USER_HOST" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Opening SSH master to $DEST_USER_HOST (enter password once) ==="
ssh -M -S "$SOCK" -fN "$DEST_USER_HOST"

RSH="ssh -S $SOCK"

echo ""
echo "=== Ensuring destination folder exists: $DEST_PATH ==="
ssh -S "$SOCK" "$DEST_USER_HOST" "mkdir -p '$DEST_PATH'"

echo ""
echo "=== Copying run $RUN from $FOLDER ==="
echo "  src:  $SRC_PATH/"
echo "  dest: $DEST_USER_HOST:$DEST_PATH/"
rsync -avh --partial --progress -e "$RSH" "$SRC_PATH/" "$DEST_USER_HOST:$DEST_PATH/"

echo ""
echo "=== Done: $FOLDER/$RUN copied to iris-ws-7 ==="
