#!/bin/sh
#SBATCH --nodes=1
#SBATCH -A nvr_rtr_rgbxnext
#SBATCH --gres gpu:2
#SBATCH --partition=batch_singlenode
#SBATCH -t 03:59:00
#SBATCH --signal=USR1@100
#SBATCH --output=outputs/slurm/%j/logs.log  # Output file inside the job-specific folder

mkdir -p outputs/slurm/$SLURM_JOB_ID

script_name=$1
gpu_query="nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv"
init="source /opt/miniconda/bin/activate neugen"

PROJ=$DISK/neugen

# Run the actual training code inside the Pytorch container
srun --container-image $DISK/cache/neugen.sqsh \
     --container-mounts=$DISK/neugen:/home/NeuGen,/lustre:/lustre \
     bash -c "$gpu_query && $init && cd $PROJ && bash scripts/$script_name.sh out_dir='\${exp_name}' out_root='outputs/slurm/$SLURM_JOB_ID/\${exp_name}'"

# # Read the status
# # Set retry limit
# MAX_RETRIES=3

# # Define file paths
# RETRY_COUNT_FILE="status/${SLURM_JOB_ID}_retry_count.txt"

# # Initialize retry count if not already done
# if [ ! -f "${RETRY_COUNT_FILE}" ]; then
#     echo "Retry file (${RETRY_COUNT_FILE}) not found"
#     echo 0 > "${RETRY_COUNT_FILE}"
# fi

# # Read the current retry count
# CURRENT_RETRIES=$(cat "${RETRY_COUNT_FILE}")

# # Check if retry limit is exceeded
# if [ "${CURRENT_RETRIES}" -ge "${MAX_RETRIES}" ]; then
#     echo "Retry limit (${MAX_RETRIES}) reached. Exiting..."
#     exit 1
# fi

# # Check if the status file exists at outputs/${SLURM_JOB_ID}/status.txt
# STATUS_FILE="status/${SLURM_JOB_ID}.txt"
# if [ ! -f ${STATUS_FILE} ]; then
#     echo "Status file not found. Exiting..."
#     exit 1
# else
#     echo "Status file found"
# fi

# # The status file (SLURM_JOB_ID.txt) will be updated with the current status of the job.
# # If it is STOPPED or RUNNING, resubmit the job to continue training.

# STATUS=$(cat $STATUS_FILE)
# # If the status is STOPPED or RUNNING, resubmit the job
# if [ "${STATUS}" == "STOPPED" ] || [ "${STATUS}" == "RUNNING" ]; then
#     echo "Requeuing job. Current attempt: $((CURRENT_RETRIES + 1)) of ${MAX_RETRIES}."
#     # Increment and save the retry count
#     echo $((CURRENT_RETRIES + 1)) > "${RETRY_COUNT_FILE}"
#     scontrol requeue $SLURM_JOB_ID
# else
#     echo "Job finished with status: ${STATUS}"
# fi
