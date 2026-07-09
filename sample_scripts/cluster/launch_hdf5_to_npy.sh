#!/bin/bash
# Usage: ./launch_hdf5_to_npy.sh <TOTAL_MATERIALS> [ARRAY_SPEC] [HDF5_DIR] [OUT_DIR]
# Example: ./launch_hdf5_to_npy.sh 5675 0-99
#          ./launch_hdf5_to_npy.sh 5675 0-99%10  # at most 10 array tasks at a time (Slurm %N)
#          With FILE_LIST=ost_greedy.txt: same 57 materials per task, greedy order, max 10 concurrent.
#
# Optional: export FILE_LIST=ost_greedy.txt (or ost_order.txt) for OST-friendly file order.
# For disjoint OSTs: export FILE_LIST and DISJOINT_RANGES, use array 0-(K-1).
#
# Submits a Slurm array job: each task converts a chunk of materials.
# Partition: cpu_short, 10 CPUs per task.

set -e
cd "$(dirname "$0")/../.."

TOTAL_MATERIALS="${1:?Usage: $0 <TOTAL_MATERIALS> [ARRAY_SPEC] [HDF5_DIR] [OUT_DIR]}"
ARRAY_SPEC="${2:-0-99}"
HDF5_DIR="${3:-/lustre/fsw/portfolios/nvr/users/shadadan/hd5}"
OUT_DIR="${4:-/lustre/fsw/portfolios/nvr/users/shadadan/hd5_sep}"

LOG_BASE="/lustre/fsw/portfolios/nvr/users/shadadan/neugen/outputs/slurm/hdf5_to_npy"
mkdir -p "$LOG_BASE"
NEUGEN_DIR="$(pwd)"
# Use OST-ordered list if present (from: python scripts/hdf5_ost_analysis.py ... -o ost_order.txt)
FILE_LIST="${FILE_LIST:-}"
[ -n "$FILE_LIST" ] && [ ! -f "$FILE_LIST" ] && FILE_LIST="$NEUGEN_DIR/$FILE_LIST"
DISJOINT_RANGES="${DISJOINT_RANGES:-}"
[ -n "$DISJOINT_RANGES" ] && [ ! -f "$DISJOINT_RANGES" ] && DISJOINT_RANGES="$NEUGEN_DIR/$DISJOINT_RANGES"
# Conda on compute nodes: set before launch if needed, e.g.:
#   export CONDA_ACTIVATE='source $HOME/miniconda3/bin/activate neugen'
JOB_ID=$(sbatch -A nvr_rtr_difftracer \
    --partition=cpu_short \
    --array="$ARRAY_SPEC" \
    --export=ALL,TOTAL_MATERIALS=$TOTAL_MATERIALS,HDF5_DIR=$HDF5_DIR,OUT_DIR=$OUT_DIR,NEUGEN_DIR=$NEUGEN_DIR,FILE_LIST=$FILE_LIST,DISJOINT_RANGES=$DISJOINT_RANGES,CONDA_ACTIVATE \
    --parsable \
    sample_scripts/cluster/hdf5_to_npy.sh)
echo "HDF5->NPY conversion job: $JOB_ID"
echo "  Total materials: $TOTAL_MATERIALS  Array: $ARRAY_SPEC"
echo "  Input:  $HDF5_DIR"
echo "  Output: $OUT_DIR"
echo "  Logs:   $LOG_BASE/${JOB_ID}/<task_id>.log"
[ -n "$FILE_LIST" ] && echo "  File list: $FILE_LIST"
[ -n "$DISJOINT_RANGES" ] && echo "  Disjoint ranges: $DISJOINT_RANGES"
