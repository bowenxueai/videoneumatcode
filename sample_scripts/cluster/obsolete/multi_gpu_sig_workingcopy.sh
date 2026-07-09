#!/bin/sh

#SBATCH --nodes=1
#SBATCH -A nvr_rtr_rgbxnext
#SBATCH --gres gpu:2
#SBATCH --partition=polar4
#SBATCH -t 03:59:00
#SBATCH --signal=B:USR1@100
#SBATCH --output=outputs/slurm/%j/logs.log

mkdir -p outputs/slurm/$SLURM_JOB_ID

# Clean up any old termination marker from previous run (important for requeued jobs!)
TERMINATE_MARKER_CLEANUP="outputs/slurm/$SLURM_JOB_ID/TERMINATE_REQUESTED"
if [ -f "$TERMINATE_MARKER_CLEANUP" ]; then
    echo "[sbatch] Removing old termination marker from previous run"
    rm -f "$TERMINATE_MARKER_CLEANUP"
fi

echo $SLURM_NNODES $SLURM_GPUS_ON_NODE

script_name=$1

gpu_query='echo GPUS_ON_NODE=$SLURM_GPUS_ON_NODE && nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv'
init="source /opt/miniconda/bin/activate neugen"

acc_init=$(cat <<'EOF'
export HF_HOME="/opt"
mkdir -p "$HF_HOME/accelerate"
printf "distributed_type: MULTI_GPU\nnum_machines: 1\nnum_processes: %s\n" "${SLURM_GPUS_ON_NODE}" > "$HF_HOME/accelerate/default_config.yaml"
EOF
)

PROJ=$DISK/neugen

# Run training with signal forwarding wrapper
# Install traps in the batch shell and forward to the srun step PID (similar to ADLR)
# File-based signal marker - more reliable than process signals for multi-GPU training
TERMINATE_MARKER="outputs/slurm/$SLURM_JOB_ID/TERMINATE_REQUESTED"

# Handler for SIGUSR1 (SLURM timeout) - save checkpoint before exit
sigusr1_handler() {
    echo "[sbatch] SIGUSR1 (timeout preemption) caught. Creating termination marker for checkpoint save."
    # Create marker file so Python saves checkpoint
    touch "$TERMINATE_MARKER"
    echo "SIGUSR1" > "$TERMINATE_MARKER"
    # DO NOT forward signal to srun - srun will kill processes immediately
    # Instead, just wait for Python to detect the file marker and exit gracefully
    echo "[sbatch] Waiting for Python to detect marker and save checkpoint..."
    wait "$srun_pid"
    status_from_sig=$?
    echo "[sbatch] Signal handler complete. srun exit after SIGUSR1: $status_from_sig"
}

# Handler for SIGTERM (manual scancel) - exit immediately, NO checkpoint
sigterm_handler() {
    echo "[sbatch] SIGTERM (manual cancel) caught. Exiting WITHOUT saving checkpoint."
    # Do NOT create termination marker - we don't want checkpoint saved
    # Kill srun immediately
    kill -s TERM "$srun_pid" 2>/dev/null
    wait "$srun_pid"
    status_from_sig=$?
    echo "[sbatch] Signal handler complete. srun exit after SIGTERM: $status_from_sig"
}

trap 'sigusr1_handler' USR1
trap 'sigterm_handler' TERM

srun --container-image $DISK/cache/neugen.sqsh \
     --container-mounts=$DISK/neugen:/home/NeuGen,/lustre:/lustre \
     bash -c "
# Setup signal forwarding inside the container

# SIGUSR1 (timeout) - DO NOT forward to child!
# PyTorch elastic will kill workers if it receives the signal.
# Rely on the file marker for graceful checkpoint save.
handle_usr1() {
    echo '[container] SIGUSR1 received (timeout). NOT forwarding - relying on file marker for graceful save.'
    # Just wait - Python workers will detect the TERMINATE_REQUESTED file and save
}

# SIGTERM (manual cancel) - forward as SIGTERM for immediate exit, no checkpoint
handle_term() {
    echo '[container] SIGTERM received (manual cancel), terminating child immediately'
    kill -TERM \$child_pid 2>/dev/null
}

trap 'handle_usr1' USR1
trap 'handle_term' TERM INT

$gpu_query && $init && $acc_init && cat /opt/accelerate/default_config.yaml && cd $PROJ

# Export the terminate marker path so Python can check it
export SLURM_TERMINATE_MARKER='$PROJ/outputs/slurm/$SLURM_JOB_ID/TERMINATE_REQUESTED'

# Run the training script in background so we can forward signals
bash scripts/$script_name.sh out_dir='\${exp_name}' out_root='outputs/slurm/$SLURM_JOB_ID/\${exp_name}' &
child_pid=\$!
echo [container] Child PID: \$child_pid

# Wait loop that allows signal handling
while kill -0 \$child_pid 2>/dev/null; do
    wait \$child_pid 2>/dev/null || true
done
wait \$child_pid 2>/dev/null
exit \$?
" &
srun_pid=$!
echo "[sbatch] Launched srun step. PID=$srun_pid"

# Wait for srun to complete (either normally or due to forwarded signal)
wait "$srun_pid"
status=$?
echo "[sbatch] srun completed. Exit status=$status"

# Find the output directory and status file
OUTPUT_DIR=$(find outputs/slurm/${SLURM_JOB_ID}/ -maxdepth 1 -type d ! -path "outputs/slurm/${SLURM_JOB_ID}/" 2>/dev/null | head -n 1)
STATUS_FILE="${OUTPUT_DIR}/status.txt"

echo "Output dir: ${OUTPUT_DIR}"
echo "Status file: ${STATUS_FILE}"

# Check job status - only two valid states: "finished" and "requeue"
if [ -f "${STATUS_FILE}" ]; then
    STATUS=$(cat "${STATUS_FILE}")
    echo "Job status: ${STATUS}"
    
    case "${STATUS}" in
        "finished")
            echo "Job completed successfully. Exiting."
            exit 0
            ;;
        "requeue")
            echo "Job received timeout signal and saved checkpoint. Requeuing..."
            ;;
        *)
            echo "Unexpected status: ${STATUS}. Not requeuing."
            exit 1
            ;;
    esac
else
    echo "No status file found (job crashed, error, or manual cancel). Not requeuing."
    exit 1
fi

# Retry logic
MAX_RETRIES=5
RETRY_COUNT_FILE="outputs/slurm/${SLURM_JOB_ID}/_retry_count.txt"

if [ ! -f "${RETRY_COUNT_FILE}" ]; then
    echo 0 > "${RETRY_COUNT_FILE}"
fi

CURRENT_RETRIES=$(cat "${RETRY_COUNT_FILE}")

if [ "${CURRENT_RETRIES}" -ge "${MAX_RETRIES}" ]; then
    echo "Retry limit (${MAX_RETRIES}) reached. Exiting."
    exit 1
fi

# Save current log and requeue
cp outputs/slurm/$SLURM_JOB_ID/logs.log outputs/slurm/$SLURM_JOB_ID/logs_${CURRENT_RETRIES}.log 2>/dev/null || true
echo "Requeuing job. Attempt $((CURRENT_RETRIES + 1)) of ${MAX_RETRIES}."
echo $((CURRENT_RETRIES + 1)) > "${RETRY_COUNT_FILE}"
scontrol requeue $SLURM_JOB_ID
