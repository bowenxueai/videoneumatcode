#!/bin/sh

#SBATCH --nodes=2
#SBATCH -A nvr_rtr_rgbxnext
#SBATCH --gres gpu:2
#SBATCH --partition=polar,polar3,polar4
#SBATCH -t 03:59:00
#SBATCH --signal=B:USR1@500
#SBATCH --output=outputs/slurm/23202477/logs.log

# Usage: sbatch multi_node_sig.sh <script_name> [max_retries]
# max_retries: 0 = no retries (default), -1 = unlimited, N = max N retries
# Example: sbatch --nodes=4 multi_node_sig.sh run1 5

script_name=$1
MAX_RETRIES=${2:-0}  # Default: 0 (no retries)

# Get head node IP for distributed coordination
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
FOLDER=$SLURM_JOB_ID
FOLDER='23202477'

mkdir -p outputs/slurm/$FOLDER


echo "Nodes: $SLURM_NNODES, GPUs per node: $SLURM_GPUS_ON_NODE, Head node: $head_node_ip, Max retries: $MAX_RETRIES"

gpu_query='echo GPUS_ON_NODE=$SLURM_GPUS_ON_NODE SLURM_NNODES=$SLURM_NNODES && nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv'
init="source /opt/miniconda/bin/activate neugen"

acc_init=$(cat <<'EOF'
export HF_HOME="/opt"
mkdir -p "$HF_HOME/accelerate"
TOTAL_PROCS="$((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
PORT=$((20000 + (SLURM_JOB_ID % 10000)))
printf "distributed_type: MULTI_GPU\nnum_machines: %s\nnum_processes: %s\nmachine_rank: %s\nmain_process_ip: %s\nmain_process_port: %s\n" \
  "${SLURM_NNODES}" "${TOTAL_PROCS}" "${SLURM_PROCID}" "${MAIN_IP}" "${PORT}" > "$HF_HOME/accelerate/default_config.yaml"
EOF
)

PROJ=$DISK/neugen

# Run training (srun in background so trap can fire)
srun --container-image $DISK/cache/neugen.sqsh \
     --container-mounts=$DISK/neugen:/home/NeuGen,/lustre:/lustre \
     bash -c "
# Trap SIGUSR1 inside container - ignore it, let Python detect file marker instead
export MAIN_IP='${head_node_ip}'
$gpu_query && $init && $acc_init && cat /opt/accelerate/default_config.yaml && cd $PROJ
export SLURM_TERMINATE_MARKER='$PROJ/outputs/slurm/$FOLDER/TERMINATE_REQUESTED'
bash scripts/$script_name.sh out_dir='\${exp_name}' out_root='outputs/slurm/$FOLDER/\${exp_name}'
" &
srun_pid=$!

# Wait for training to complete
wait "$srun_pid"
echo "[sbatch] srun completed. Exit status=$?"

# Find the output directory and status file
OUTPUT_DIR=$(find outputs/slurm/${FOLDER}/ -maxdepth 1 -type d ! -path "outputs/slurm/${FOLDER}/" 2>/dev/null | head -n 1)

echo "Output dir: ${OUTPUT_DIR}"

# Retry logic (only reached if STATUS = "requeue")
if [ "${MAX_RETRIES}" -eq 0 ]; then
    echo "Retries disabled (MAX_RETRIES=0). Exiting."
    exit 0
fi

RETRY_COUNT_FILE="outputs/slurm/${FOLDER}/_retry_count.txt"

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
cp outputs/slurm/$FOLDER/logs.log outputs/slurm/$FOLDER/logs_${CURRENT_RETRIES}.log 2>/dev/null || true
if [ "${MAX_RETRIES}" -lt 0 ]; then
    echo "Requeuing job. Attempt $((CURRENT_RETRIES + 1)) (unlimited)."
else
    echo "Requeuing job. Attempt $((CURRENT_RETRIES + 1)) of ${MAX_RETRIES}."
fi
echo $((CURRENT_RETRIES + 1)) > "${RETRY_COUNT_FILE}"
scontrol requeue $SLURM_JOB_ID

