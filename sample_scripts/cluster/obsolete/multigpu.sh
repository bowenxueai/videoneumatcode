#!/bin/sh

#SBATCH --nodes=1
#SBATCH -A nvr_rtr_rgbxnext
#SBATCH --gres gpu:2
#SBATCH --partition=polar4
#SBATCH -t 03:59:00
#SBATCH --signal=B:USR1@100
#SBATCH --output=outputs/slurm/%j/logs.log

mkdir -p outputs/slurm/$SLURM_JOB_ID

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
# Run the actual training code inside the Pytorch container
srun --container-image $DISK/cache/neugen.sqsh \
     --container-mounts=$DISK/neugen:/home/NeuGen,/lustre:/lustre \
     bash -c "$gpu_query && $init && $acc_init && cat /opt/accelerate/default_config.yaml && cd $PROJ && bash scripts/$script_name.sh out_dir='\${exp_name}' out_root='outputs/slurm/$SLURM_JOB_ID/\${exp_name}'"

# Find the output directory and status file
OUTPUT_DIR=$(find outputs/slurm/${SLURM_JOB_ID}/ -maxdepth 1 -type d ! -path "outputs/slurm/${SLURM_JOB_ID}/" 2>/dev/null | head -n 1)
STATUS_FILE="${OUTPUT_DIR}/status.txt"

echo "Output dir: ${OUTPUT_DIR}"
echo "Status file: ${STATUS_FILE}"

# Check job status
if [ -f "${STATUS_FILE}" ]; then
    STATUS=$(cat "${STATUS_FILE}")
    echo "Job status: ${STATUS}"
    
    case "${STATUS}" in
        "finished")
            echo "Job completed successfully. Exiting."
            exit 0
            ;;
        "requeue")
            echo "Job received preemption signal and saved state. Requeuing..."
            ;;
        "started")
            echo "Job crashed without clean exit. Attempting requeue..."
            ;;
        "failed")
            echo "Job failed with error. Not requeuing."
            exit 1
            ;;
        *)
            echo "Unknown status: ${STATUS}. Not requeuing."
            exit 1
            ;;
    esac
else
    echo "No status file found (Hydra may have failed or job crashed very early). Not requeuing."
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
