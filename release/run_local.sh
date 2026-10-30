#!/bin/bash
# ============================================================================
# VideoNeuMat — run the WHOLE pipeline LOCALLY (no SLURM, no container), on one
# local GPU. Same result as oneclick_material.sh, just run inline.
#
# Needs two Python envs (build from init/, see README):
#   GEN_ENV     : generation deps (init/gen_requirements.txt)          [Stage A]
#   NEUGEN_ENV  : NeuGen / LRM deps (init/environment.yml, incl h5py)  [Stage B]
# and the weights (release/download_weights.sh) with config.env pointing at them.
#
# Usage:
#   GEN_ENV=/path/gen  NEUGEN_ENV=/path/neugen  release/run_local.sh "a material prompt"
#   ... release/run_local.sh --lora "a prompt"
#   ... release/run_local.sh --i2v material.jpg "a prompt"
# ============================================================================
set -euo pipefail
RELEASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RELEASE_DIR
source "${RELEASE_DIR}/config.env"
: "${NEUGEN_ENV:?set NEUGEN_ENV to a python env dir (with bin/python) that has the NeuGen deps}"

PROMPT="" ; RUNNAME="" ; INPUT_IMAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --i2v)  INPUT_IMAGE="$2"; shift 2 ;;
    --lora) export GEN_MODE=lora MATERIAL_CKPT="${WEIGHTS_DIR}/videoneumat/t2v_lora/videoneumat-t2v-lora.safetensors"; shift ;;
    --name) RUNNAME="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) PROMPT="$1"; shift ;;
  esac
done
[ -n "$PROMPT" ] || { echo "ERROR: give a prompt (add --i2v IMAGE for image->video). Try --help." >&2; exit 1; }
[ -n "$INPUT_IMAGE" ] && [ ! -f "$INPUT_IMAGE" ] && { echo "ERROR: --i2v image not found: $INPUT_IMAGE" >&2; exit 1; }
[ -n "$INPUT_IMAGE" ] && INPUT_IMAGE="$(realpath -e -- "$INPUT_IMAGE")"

STAMP="$(date +%Y%m%d%H%M%S)" ; SPLIT="rel${STAMP}"
[ -n "$RUNNAME" ] && SPLIT="rel$(echo "$RUNNAME" | tr -cd '[:alnum:]')${STAMP}"
RUN_DIR="${NEUGEN_REPO}/outputs/release_runs/${SPLIT}" ; GEN_OUTDIR="${RUN_DIR}/videos"
mkdir -p "$GEN_OUTDIR" ; export RUN_DIR SPLIT GEN_OUTDIR
echo "run: $SPLIT  ($([ -n "$INPUT_IMAGE" ] && echo I2V || echo "T2V ${GEN_MODE}")) — LOCAL (no SLURM)"

cd "$NEUGEN_REPO"
export PYTHONPATH="$NEUGEN_REPO${PYTHONPATH:+:$PYTHONPATH}"   # vendored ./diffsynth

# ---- Stage A: generate the material video (GEN_ENV) ------------------------
export DIFFSYNTH_MODEL_BASE_PATH="$WAN_MODELS" DIFFSYNTH_SKIP_DOWNLOAD=true TOKENIZERS_PARALLELISM=false
if [ -n "$INPUT_IMAGE" ]; then
  "$GEN_ENV/bin/python" "$RELEASE_DIR/gen_i2v.py" \
    --ckpt "$I2V_CKPT" --image "$INPUT_IMAGE" --prompt "$PROMPT" --out "$GEN_OUTDIR/0000.mp4" \
    --base_id "$I2V_BASE_ID" --num_frames "$NUM_FRAMES" --height "$RES" --width "$RES" \
    --steps "$STEPS" --cfg_scale "$CFG" --seed "$SEED"
else
  printf 'prompt\n' > "$RUN_DIR/prompts.csv"
  printf '%s\n' "\"${PROMPT//\"/\"\"}\"" >> "$RUN_DIR/prompts.csv"
  "$GEN_ENV/bin/python" "$RELEASE_DIR/gen_material_video.py" \
    --mode "$GEN_MODE" --ckpt "$MATERIAL_CKPT" --prompts_csv "$RUN_DIR/prompts.csv" --out_dir "$GEN_OUTDIR" \
    --num_frames "$NUM_FRAMES" --height "$RES" --width "$RES" --steps "$STEPS" --cfg_scale "$CFG" \
    --seed "$SEED" --neg_style "$NEG_STYLE" --pos_style none
fi

# ---- Stage B: extract the neural material (NEUGEN_ENV) ---------------------
NEUGEN_PY="$NEUGEN_ENV/bin/python" bash "$RELEASE_DIR/extract_core.sh"
