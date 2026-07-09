#!/bin/bash

#SBATCH -A nvr_rtr_rgbxnext
#SBATCH --partition=polar,polar3,polar4
#SBATCH --nodes=8                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres gpu:8                # number of GPUs per node
#SBATCH --time 03:59:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=outputs/slurm/%j/logs.log  # Output file inside the job-specific folder

######################
### Set environment ###
######################
export GPUS_PER_NODE=8
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################
PROJ=$DISK/neugen
echo hello
echo $SLURM_NNODES $GPUS_PER_NODE
export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank \$SLURM_PROCID \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --mixed_precision bf16 \
    "

export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
export SCRIPT="src/inverse_render_LRM.py"
export ARGS="batch_size=4 \
    resolution=256 \
    dataset@train_dataset=[matsynth_all_but] \
    checkpoint.network=/lustre/fsw/portfolios/nvr/users/shadadan/neugen/outputs/good/standalone-matsynth512-256-offset_True-TM_srgb_no_clamp-net_l12-64x4-bs32/model.pth \
    n_iterations=900"
    

gpu_query="nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv"
init="source /opt/miniconda/bin/activate neugen"

# This step is necessary because accelerate launch does not handle multiline arguments properly
# export CMD=" $LAUNCHER $SCRIPT $ARGS" 
srun --container-image $DISK/cache/neugen.sqsh --container-mounts=$DISK/neugen:/home/NeuGen,/lustre:/lustre bash -c "$init && cd $PROJ && $LAUNCHER $SCRIPT $ARGS"


# mkdir -p outputs/slurm/$SLURM_JOB_ID

# script_name=$1
# gpu_query="nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv"
# init="source /opt/miniconda/bin/activate neugen"

# PROJ=$DISK/neugen

# # Run the actual training code inside the Pytorch container
# srun --container-image $DISK/cache/neugen.sqsh \
#      --container-mounts=$DISK/neugen:/home/NeuGen,/lustre:/lustre \
#      bash -c "$gpu_query && $init && cd $PROJ && bash scripts/$script_name.sh out_dir='\${exp_name}' out_root='outputs/slurm/$SLURM_JOB_ID/\${exp_name}'"
