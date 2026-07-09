#!/bin/bash
#SBATCH --nodes=1
#SBATCH -A nvr_rtr_rgbxnext
#SBATCH --gres gpu:1
#SBATCH --partition=polar,polar3,polar4
#SBATCH -t 03:59:00
#SBATCH --output=outputs/slurm/render_btf/%A/%a.log
#
# Usage: sbatch --array=0-7 render_btf.sh
#        (for 8 GPUs, use --array=0-7)
#        (for 16 GPUs, use --array=0-15)
#
# Total materials set via TOTAL_MATERIALS env var or defaults to 89
# Each array task processes TOTAL_MATERIALS/NUM_TASKS materials

TOTAL_MATERIALS="${TOTAL_MATERIALS:-89}"
RESOLUTION="${RESOLUTION:-1024}"
TRAJ="${TRAJ:-sparse}"

# Get number of tasks from array size
NUM_TASKS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
NUM_TASKS=300
TASK_ID=$SLURM_ARRAY_TASK_ID

# Calculate materials per task with even distribution of remainder
MATERIALS_PER_TASK=$((TOTAL_MATERIALS / NUM_TASKS))
REMAINDER=$((TOTAL_MATERIALS % NUM_TASKS))

if [ $TASK_ID -lt $REMAINDER ]; then
    MAX_MATERIALS=$((MATERIALS_PER_TASK + 1))
    START_IDX=$((TASK_ID * (MATERIALS_PER_TASK + 1)))
else
    MAX_MATERIALS=$MATERIALS_PER_TASK
    START_IDX=$((REMAINDER * (MATERIALS_PER_TASK + 1) + (TASK_ID - REMAINDER) * MATERIALS_PER_TASK))
fi


FOLDER=$SLURM_ARRAY_JOB_ID
FOLDER=23054397
mkdir -p outputs/slurm/render_btf/$FOLDER

echo "[sbatch] JOB=$SLURM_ARRAY_JOB_ID (FOLDER=$FOLDER) TASK=$TASK_ID START=$START_IDX COUNT=$MAX_MATERIALS RES=$RESOLUTION TRAJ=$TRAJ"

gpu_query='echo "[container]: GPUS_ON_NODE=$SLURM_GPUS_ON_NODE" && nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv'
init="source /opt/miniconda/bin/activate neugen"

PROJ=$DISK/neugen

# Run the render command inside the container
srun --container-image $DISK/cache/neugen.sqsh \
     --container-mounts=$DISK/neugen:/home/NeuGen,/lustre:/lustre,/lustre/fsw/portfolios/nvr/users/shadadan/nvoptix.bin:/usr/share/nvidia/nvoptix.bin \
     bash -c "$gpu_query && $init && cd $PROJ && python src/render_haha_2.py --max-materials $MAX_MATERIALS --start-idx $START_IDX --num-gpus 1 -r $RESOLUTION --traj $TRAJ --save-frames --output outputs/slurm/render_btf/${FOLDER}/${SLURM_ARRAY_TASK_ID}"

