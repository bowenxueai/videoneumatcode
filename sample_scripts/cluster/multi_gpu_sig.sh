#!/bin/sh

#SBATCH --nodes=1
#SBATCH -A nvr_rtr_rgbxnext
#SBATCH --gres gpu:2
#SBATCH --partition=polar,polar3,polar4
#SBATCH -t 03:59:00
#SBATCH --signal=B:USR1@500
#SBATCH --output=outputs/slurm/%j/logs.log

# Usage: sbatch multi_gpu_sig.sh <script_name> [max_retries]
# max_retries: 0 = no retries (default), -1 = unlimited, N = max N retries
# Example: sbatch multi_gpu_sig.sh run1 5

script_name=$1
MAX_RETRIES=${2:-0}  # Default: 0 (no retries)

mkdir -p outputs/slurm/$SLURM_JOB_ID

# Clean up stale files from previous run (important for requeued jobs!)
TERMINATE_MARKER="outputs/slurm/$SLURM_JOB_ID/TERMINATE_REQUESTED"
if [ -f "$TERMINATE_MARKER" ]; then
    echo "[sbatch] Removing old termination marker from previous run"
    rm -f "$TERMINATE_MARKER"
fi

# Clear old status file to prevent infinite requeue loop if this run crashes
OLD_STATUS_FILE=$(find outputs/slurm/${SLURM_JOB_ID}/ -name "status.txt" 2>/dev/null | head -n 1)
if [ -n "$OLD_STATUS_FILE" ] && [ -f "$OLD_STATUS_FILE" ]; then
    echo "[sbatch] Removing old status file: $OLD_STATUS_FILE"
    rm -f "$OLD_STATUS_FILE"
fi

echo "Nodes: $SLURM_NNODES, GPUs per node: $SLURM_GPUS_ON_NODE, Max retries: $MAX_RETRIES"

gpu_query='echo GPUS_ON_NODE=$SLURM_GPUS_ON_NODE && nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv'
init="source /opt/miniconda/bin/activate neugen"

acc_init=$(cat <<'EOF'
export HF_HOME="/opt"
mkdir -p "$HF_HOME/accelerate"
printf "distributed_type: MULTI_GPU\nnum_machines: 1\nnum_processes: %s\n" "${SLURM_GPUS_ON_NODE}" > "$HF_HOME/accelerate/default_config.yaml"
EOF
)

PROJ=$DISK/neugen

# Handler for SIGUSR1 (SLURM timeout) - create marker and wait for graceful exit
sigusr1_handler() {
    echo "[sbatch] SIGUSR1 received. Creating termination marker..."
    touch "$TERMINATE_MARKER"
    # Wait inside handler so main script doesn't proceed until srun finishes
    echo "[sbatch] Waiting for training to save checkpoint and exit..."
    wait "$srun_pid"
    echo "[sbatch] srun completed after SIGUSR1. Exit status=$?"
}
trap 'sigusr1_handler' USR1

# Run training (srun in background so trap can fire)
srun --container-image $DISK/cache/neugen.sqsh \
     --container-mounts=$DISK/neugen:/home/NeuGen,/lustre:/lustre \
     bash -c "
# Trap SIGUSR1 inside container - ignore it, let Python detect file marker instead
trap '' USR1

$gpu_query && $init && $acc_init && cat /opt/accelerate/default_config.yaml && cd $PROJ
export SLURM_TERMINATE_MARKER='$PROJ/outputs/slurm/$SLURM_JOB_ID/TERMINATE_REQUESTED'
bash scripts/$script_name.sh out_dir='\${exp_name}' out_root='outputs/slurm/$SLURM_JOB_ID/\${exp_name}'
" &
srun_pid=$!

# Wait for training to complete
wait "$srun_pid"
echo "[sbatch] srun completed. Exit status=$?"

# Find the output directory and status file
OUTPUT_DIR=$(find outputs/slurm/${SLURM_JOB_ID}/ -maxdepth 1 -type d ! -path "outputs/slurm/${SLURM_JOB_ID}/" 2>/dev/null | head -n 1)
STATUS_FILE="${OUTPUT_DIR}/status.txt"

echo "Output dir: ${OUTPUT_DIR}"
echo "Status file: ${STATUS_FILE}"

# Check job status - only "requeue" triggers requeue
if [ -f "${STATUS_FILE}" ]; then
    STATUS=$(cat "${STATUS_FILE}")
    echo "Job status: ${STATUS}"
    
    if [ "${STATUS}" = "requeue" ]; then
        echo "Job saved checkpoint. Requeuing..."
    elif [ "${STATUS}" = "finished" ]; then
        echo "Job completed successfully. Exiting."
        exit 0
    else
        echo "Unexpected status: ${STATUS}. Not requeuing."
        exit 1
    fi
else
    echo "No status file found. Not requeuing."
    exit 1
fi

# Retry logic (only reached if STATUS = "requeue")
if [ "${MAX_RETRIES}" -eq 0 ]; then
    echo "Retries disabled (MAX_RETRIES=0). Exiting."
    exit 0
fi

RETRY_COUNT_FILE="outputs/slurm/${SLURM_JOB_ID}/_retry_count.txt"

if [ ! -f "${RETRY_COUNT_FILE}" ]; then
    echo 0 > "${RETRY_COUNT_FILE}"
fi

CURRENT_RETRIES=$(cat "${RETRY_COUNT_FILE}")

# Check retry limit (skip if MAX_RETRIES=-1 for unlimited)
if [ "${MAX_RETRIES}" -gt 0 ] && [ "${CURRENT_RETRIES}" -ge "${MAX_RETRIES}" ]; then
    echo "Retry limit (${MAX_RETRIES}) reached. Exiting."
    exit 1
fi

# Save current log and requeue
cp outputs/slurm/$SLURM_JOB_ID/logs.log outputs/slurm/$SLURM_JOB_ID/logs_${CURRENT_RETRIES}.log 2>/dev/null || true
if [ "${MAX_RETRIES}" -lt 0 ]; then
    echo "Requeuing job. Attempt $((CURRENT_RETRIES + 1)) (unlimited)."
else
    echo "Requeuing job. Attempt $((CURRENT_RETRIES + 1)) of ${MAX_RETRIES}."
fi
echo $((CURRENT_RETRIES + 1)) > "${RETRY_COUNT_FILE}"
scontrol requeue $SLURM_JOB_ID
