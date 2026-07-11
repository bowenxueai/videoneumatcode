#!/bin/bash
# Stage A (I2V variant) — generate an 81f/1024 material video from a still image with the
# fine-tuned Wan2.1-I2V-14B DiT. Produces $GEN_OUTDIR/0000.mp4 (same as the T2V gen), so the
# unchanged extract_material.sbatch turns it into a neural material.
# Needs (via --export=ALL): RELEASE_DIR, GEN_OUTDIR, PROMPT, INPUT_IMAGE, I2V_CKPT (+ config.env).
# WAN_MODELS must hold Wan-AI/Wan2.1-I2V-14B-720P and Wan-AI/Wan2.1-T2V-1.3B (tokenizer).
#SBATCH --job-name=neugen_gen_i2v
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
: "${RELEASE_DIR:?}"
source "${RELEASE_DIR}/config.env"
: "${GEN_OUTDIR:?}" ; : "${PROMPT:?}" ; : "${INPUT_IMAGE:?}" ; : "${I2V_CKPT:?}"
[ -f "$INPUT_IMAGE" ] || { echo "ERROR: I2V input image not found: $INPUT_IMAGE" >&2; exit 1; }
INPUT_IMAGE="$(realpath -e -- "$INPUT_IMAGE")"
export INPUT_IMAGE
mkdir -p "$GEN_OUTDIR"

export DIFFSYNTH_MODEL_BASE_PATH="$WAN_MODELS"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$NEUGEN_REPO${PYTHONPATH:+:$PYTHONPATH}"   # vendored ./diffsynth
export CUDA_HOME="$GEN_ENV"
export PATH="$GEN_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$GEN_ENV/lib:${LD_LIBRARY_PATH:-}"

cd "$NEUGEN_REPO"
echo "[gen-i2v] host=$(hostname) ckpt=$I2V_CKPT image=$INPUT_IMAGE res=${RES}x${RES}x${NUM_FRAMES}"
srun --ntasks=1 "$GEN_ENV/bin/python" "$RELEASE_DIR/gen_i2v.py" \
  --ckpt "$I2V_CKPT" --image "$INPUT_IMAGE" --prompt "$PROMPT" --out "$GEN_OUTDIR/0000.mp4" \
  --base_id "${I2V_BASE_ID:-Wan-AI/Wan2.1-I2V-14B-720P}" \
  --num_frames "$NUM_FRAMES" --height "$RES" --width "$RES" \
  --steps "$STEPS" --cfg_scale "$CFG" --seed "$SEED"
echo "[gen-i2v] DONE -> $GEN_OUTDIR/0000.mp4"
