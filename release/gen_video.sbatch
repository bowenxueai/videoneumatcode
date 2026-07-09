#!/bin/bash
# ============================================================================
# Stage A — generate 81-frame 1024x1024 material videos with the fine-tuned
# Wan2.1-T2V-14B DiT, using the vendored diffsynth library (./diffsynth).
# One array task == one prompt == one GPU == one mp4 (~70 min on an 80GB GPU).
#
# Driven by release/oneclick_material.sh, which passes (via --export=ALL):
#   RELEASE_DIR, PROMPTS_CSV, GEN_OUTDIR   (+ everything from config.env)
# Standalone: source config.env, then
#   RELEASE_DIR=... PROMPTS_CSV=... GEN_OUTDIR=... sbatch --array=0-<N-1> gen_video.sbatch
# ============================================================================
#SBATCH --job-name=neugen_gen
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail

: "${RELEASE_DIR:?set RELEASE_DIR (dir containing config.env)}"
source "${RELEASE_DIR}/config.env"
: "${PROMPTS_CSV:?}" ; : "${GEN_OUTDIR:?}"
mkdir -p "$GEN_OUTDIR"

# DiffSynth model resolution (Wan-AI/<repo>/<file>) is rooted at WAN_MODELS
export DIFFSYNTH_MODEL_BASE_PATH="$WAN_MODELS"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# use the VENDORED diffsynth in this repo (takes precedence over any pip-installed copy)
export PYTHONPATH="$NEUGEN_REPO${PYTHONPATH:+:$PYTHONPATH}"
# the gen env may ship its own CUDA toolkit (deepspeed import wants nvcc)
export CUDA_HOME="$GEN_ENV"
export PATH="$GEN_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$GEN_ENV/lib:${LD_LIBRARY_PATH:-}"

cd "$NEUGEN_REPO"   # run from the repo so the vendored ./diffsynth is used
echo "[gen] host=$(hostname) array=${SLURM_ARRAY_TASK_ID:-NA}/${SLURM_ARRAY_TASK_COUNT:-1} ckpt=$MATERIAL_CKPT res=${RES}x${RES}x${NUM_FRAMES} steps=$STEPS cfg=$CFG neg=$NEG_STYLE"

srun --ntasks=1 "$GEN_ENV/bin/python" "$RELEASE_DIR/gen_material_video.py" \
  --mode "$GEN_MODE" --ckpt "$MATERIAL_CKPT" \
  --prompts_csv "$PROMPTS_CSV" --out_dir "$GEN_OUTDIR" \
  --num_frames "$NUM_FRAMES" --height "$RES" --width "$RES" \
  --steps "$STEPS" --cfg_scale "$CFG" --seed "$SEED" \
  --neg_style "$NEG_STYLE" --pos_style none

echo "[gen] DONE -> $GEN_OUTDIR"
