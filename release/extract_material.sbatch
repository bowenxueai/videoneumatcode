#!/bin/bash
# ============================================================================
# Stage B (SLURM) — run extract_core.sh inside the NeuGen enroot container.
# Builds a dataset from the generated mp4(s), caches VAE latents, runs the LRM
# (optimize=False) -> neural material + novel-view / reconstruction renders.
# (For a machine without SLURM, use release/run_local.sh instead.)
#
# Driven by release/oneclick_material.sh (via --export=ALL):
#   RELEASE_DIR, RUN_DIR, SPLIT, GEN_OUTDIR   (+ everything from config.env)
# ============================================================================
#SBATCH --job-name=neugen_extract
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
set -euo pipefail

: "${RELEASE_DIR:?set RELEASE_DIR}"
source "${RELEASE_DIR}/config.env"
: "${RUN_DIR:?}" ; : "${SPLIT:?}" ; : "${GEN_OUTDIR:?}"
export RELEASE_DIR NEUGEN_REPO RUN_DIR SPLIT GEN_OUTDIR WAN_MODELS LRM_LATENT LRM_MLP

srun --ntasks=1 \
  --container-image "$NEUGEN_CONTAINER" \
  --container-mounts=/lustre:/lustre \
  --export=ALL \
  bash -c 'source /opt/miniconda/bin/activate neugen \
           && pip install -q h5py \
           && NEUGEN_PY=python bash "$RELEASE_DIR/extract_core.sh"'
