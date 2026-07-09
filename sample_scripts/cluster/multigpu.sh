#!/bin/sh
#SBATCH --nodes=1
#SBATCH -A nvr_rtr_rgbxnext
#SBATCH --gres gpu:2
#SBATCH --partition=polar,polar3,polar4
#SBATCH -t 03:59:00
#SBATCH --signal=USR1@100
#SBATCH --output=outputs/slurm/%j/logs.log  # Output file inside the job-specific folder



mkdir -p outputs/slurm/$SLURM_JOB_ID

echo "[sbatch]" GPUS_PER_NODE=$SLURM_GPUS_ON_NODE NODES=$SLURM_NNODES
script_name=$1
gpu_query='echo "[container]: GPUS_ON_NODE=$SLURM_GPUS_ON_NODE SLURM_NNODES=$SLURM_NNODES" && nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv'
init="source /opt/miniconda/bin/activate neugen"

acc_init=$(cat <<'EOF'
export HF_HOME="/opt"
mkdir -p "$HF_HOME/accelerate"
printf "distributed_type: MULTI_GPU\nnum_machines: 1\nnum_processes: %s\n" "${SLURM_GPUS_ON_NODE}" > "$HF_HOME/accelerate/default_config.yaml"
EOF
)
print_cfg="cat /opt/accelerate/default_config.yaml"


PROJ=$DISK/neugen


# Run the actual training code inside the Pytorch container
srun --container-image $DISK/cache/neugen.sqsh \
     --container-mounts=$DISK/neugen:/home/NeuGen,/lustre:/lustre \
     bash -c "$gpu_query && $init && $acc_init && $print_cfg && cd $PROJ && bash scripts/$script_name.sh out_dir='\${exp_name}' out_root='outputs/slurm/$SLURM_JOB_ID/\${exp_name}'"

