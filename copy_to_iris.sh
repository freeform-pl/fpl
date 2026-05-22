#!/bin/bash
# Copy datasets from /hai/scratch/marcelto/data to iris-ws-6.
# Mirror of copy_iris_data.sh, but pushing instead of pulling.
# Excludes any file matching *_large*.hdf5.
# Uses SSH ControlMaster so the password is only prompted ONCE.
#
# Idempotent: rsync -a skips files that already exist with the same
# size + mtime (its default delta-transfer behavior). --partial keeps
# half-copied files around so the next run resumes them instead of
# starting over. Safe to re-run anytime.

set -e

SOCK=/tmp/iris-cm-$$
DST=marcelto@iris-ws-6.stanford.edu
EXC='--exclude=*_large*.hdf5'
SRC_BASE=/hai/scratch/marcelto/data

cleanup() {
    ssh -S "$SOCK" -O exit "$DST" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Opening SSH master to $DST (enter password once) ==="
ssh -M -S "$SOCK" -fN "$DST"

RSH="ssh -S $SOCK"

# Pairs of "src_local_path  dest_remote_path"
# Trailing slash on src means "copy contents of dir into dest dir".
PAIRS=(
    "$SRC_BASE/marcelto/plate_toast_iter0_multi_qwen_1dp_iter0_2000/   /iris/u/marcelto/data/plate_toast_iter0_multi_qwen_1dp_iter0_2000/"
)

i=1
total=${#PAIRS[@]}
for pair in "${PAIRS[@]}"; do
    read -r local remote <<< "$pair"
    echo ""
    echo "=== [$i/$total] $local -> $remote ==="
    # Ensure parent dir exists on the remote side.
    parent=$(dirname "$remote")
    ssh -S "$SOCK" "$DST" "mkdir -p '$parent'"
    rsync -avh --partial --progress -e "$RSH" $EXC "$local" "$DST:$remote"
    i=$((i + 1))
done

echo ""
echo "=== All $total transfers complete ==="
