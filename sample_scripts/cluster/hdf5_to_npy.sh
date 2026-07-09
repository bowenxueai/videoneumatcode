#!/bin/bash
#SBATCH --nodes=1
#SBATCH -A nvr_rtr_rgbxnext
#SBATCH --partition=cpu_short
#SBATCH -c 10
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/shadadan/neugen/outputs/slurm/hdf5_to_npy/%A/%a.log
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/shadadan/neugen/outputs/slurm/hdf5_to_npy/%A/%a.log
#
# Usage: sbatch --array=0-99 hdf5_to_npy.sh
#        (for 100 tasks; each task processes TOTAL_MATERIALS/NUM_TASKS materials)
#
# Env: NEUGEN_DIR (required), TOTAL_MATERIALS (required unless DISJOINT_RANGES set), HDF5_DIR, OUT_DIR, FILE_LIST, DISJOINT_RANGES (optional)
# If DISJOINT_RANGES is set (from hdf5_ost_analysis.py --out-disjoint-ranges), read start_idx, count, and optional max_j from that file.
# max_j caps internal -j for that task (OSTs in component) to avoid contention; default -j 10 otherwise.
# Each array task runs: python scripts/hdf5_to_separate_npy.py ... --start-idx I --max-materials N -j J

NEUGEN_DIR="${NEUGEN_DIR:?Set NEUGEN_DIR (repo root; launcher exports it)}"
HDF5_DIR="${HDF5_DIR:-/lustre/fsw/portfolios/nvr/users/shadadan/hd5}"
OUT_DIR="${OUT_DIR:-/lustre/fsw/portfolios/nvr/users/shadadan/hd5_sep}"
FILE_LIST="${FILE_LIST:-}"
DISJOINT_RANGES="${DISJOINT_RANGES:-}"

NUM_TASKS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
TASK_ID=$SLURM_ARRAY_TASK_ID

if [ -n "$DISJOINT_RANGES" ] && [ -f "$DISJOINT_RANGES" ]; then
    # One line per task: "start_idx count [max_j]"  (max_j = OSTs in component, caps -j)
    line=$(sed -n "$((TASK_ID + 1))p" "$DISJOINT_RANGES")
    START_IDX=$(echo "$line" | awk '{print $1}')
    MAX_MATERIALS=$(echo "$line" | awk '{print $2}')
    MAX_J=$(echo "$line" | awk '{print $3}')
    if [ -z "$START_IDX" ] || [ -z "$MAX_MATERIALS" ]; then
        echo "DISJOINT_RANGES: no line for TASK_ID=$TASK_ID" >&2
        exit 1
    fi
    # Cap internal -j at OSTs in component; fallback to 1 if missing (minimize contention)
    if [ -n "$MAX_J" ] && [ "$MAX_J" -gt 0 ] 2>/dev/null; then
        JOB_JOBS=$MAX_J
    else
        JOB_JOBS=1
    fi
    # Don't exceed allocated CPUs
    CPUS="${SLURM_CPUS_PER_TASK:-8}"
    if [ "$JOB_JOBS" -gt "$CPUS" ] 2>/dev/null; then JOB_JOBS=$CPUS; fi
else
    TOTAL_MATERIALS="${TOTAL_MATERIALS:?Set TOTAL_MATERIALS (or use DISJOINT_RANGES for disjoint-OST mode)}"
    # Even distribution of materials across tasks
    MATERIALS_PER_TASK=$((TOTAL_MATERIALS / NUM_TASKS))
    REMAINDER=$((TOTAL_MATERIALS % NUM_TASKS))
    if [ "$TASK_ID" -lt "$REMAINDER" ]; then
        MAX_MATERIALS=$((MATERIALS_PER_TASK + 1))
        START_IDX=$((TASK_ID * (MATERIALS_PER_TASK + 1)))
    else
        MAX_MATERIALS=$MATERIALS_PER_TASK
        START_IDX=$((REMAINDER * (MATERIALS_PER_TASK + 1) + (TASK_ID - REMAINDER) * MATERIALS_PER_TASK))
    fi
    JOB_JOBS="${JOB_JOBS:-6}"
    CPUS="${SLURM_CPUS_PER_TASK:-8}"
    if [ "$JOB_JOBS" -gt "$CPUS" ] 2>/dev/null; then JOB_JOBS=$CPUS; fi
fi

echo "[sbatch] JOB=$SLURM_ARRAY_JOB_ID TASK=$TASK_ID START_IDX=$START_IDX MAX_MATERIALS=$MAX_MATERIALS -j $JOB_JOBS"
cd "$NEUGEN_DIR" || { echo "NEUGEN_DIR=$NEUGEN_DIR not found"; exit 1; }

# Activate conda (set CONDA_ACTIVATE in launcher if needed, e.g. 'source $HOME/miniconda3/bin/activate neugen')
if [ -n "${CONDA_ACTIVATE:-}" ]; then
    eval "$CONDA_ACTIVATE"
elif [ -f /opt/miniconda3/bin/activate ]; then
    source /opt/miniconda3/bin/activate neugen
elif [ -f /opt/miniconda/bin/activate ]; then
    source /opt/miniconda/bin/activate neugen
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate" neugen
elif [ -f "$HOME/miniconda/bin/activate" ]; then
    source "$HOME/miniconda/bin/activate" neugen
else
    echo "Conda not found. Set CONDA_ACTIVATE before launch, e.g.: export CONDA_ACTIVATE='source \$HOME/miniconda3/bin/activate neugen'"
    exit 1
fi

if [ -n "$FILE_LIST" ]; then
    python scripts/hdf5_to_separate_npy.py "$HDF5_DIR" -o "$OUT_DIR" --file-list "$FILE_LIST" \
        --start-idx "$START_IDX" \
        --max-materials "$MAX_MATERIALS" \
        -j "$JOB_JOBS"
else
    python scripts/hdf5_to_separate_npy.py "$HDF5_DIR" -o "$OUT_DIR" \
        --start-idx "$START_IDX" \
        --max-materials "$MAX_MATERIALS" \
        -j "$JOB_JOBS"
fi
