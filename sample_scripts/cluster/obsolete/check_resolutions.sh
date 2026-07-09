#!/bin/sh
#SBATCH --nodes=1
#SBATCH -A nvr_rtr_rgbxnext
#SBATCH --cpus-per-task=64
#SBATCH --partition=cpu_long
#SBATCH -t 07:59:00
#SBATCH --output=outputs/slurm/%j/check_resolutions.log

mkdir -p outputs/slurm/$SLURM_JOB_ID

echo "[sbatch] CPUS=$SLURM_CPUS_PER_TASK JOB_ID=$SLURM_JOB_ID"

init="source /opt/miniconda/bin/activate neugen"
PROJ=$DISK/neugen

srun --container-image $DISK/cache/neugen.sqsh \
     --container-mounts=$DISK/neugen:/home/NeuGen,/lustre:/lustre \
     bash -c "$init && cd $PROJ && python src/check_resolutions.py"


















