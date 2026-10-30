#!/bin/bash
# ============================================================================
# VideoNeuMat ONE-CLICK: a text prompt (or an image) -> a neural material
#
#   T2V:  prompt        --[Wan2.1-T2V-14B]--> 81f/1024 video --[LRM]--> neural material
#   I2V:  image+prompt  --[Wan2.1-I2V-14B]--> 81f/1024 video --[LRM]--> neural material
#
# Usage:
#   ./oneclick_material.sh "The video clip features a close-up view of <material>, <details>"
#   ./oneclick_material.sh --prompts_csv my_prompts.csv     # T2V batch (CSV with a 'prompt' column)
#   ./oneclick_material.sh --lora "a prompt"                # T2V with the LoRA generator
#   ./oneclick_material.sh --i2v material.jpg "a prompt"    # I2V from a still image
#   ./oneclick_material.sh --name myrun ...                 # name the run
#
# Submits two chained SLURM jobs (gen -> afterok -> extract). Prereqs & weights: see README.md
# ============================================================================
set -euo pipefail

RELEASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RELEASE_DIR
source "${RELEASE_DIR}/config.env"

# ---- args ------------------------------------------------------------------
PROMPT="" ; PROMPTS_CSV_IN="" ; RUNNAME="" ; INPUT_IMAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --prompts_csv) PROMPTS_CSV_IN="$2"; shift 2 ;;
    --i2v)         INPUT_IMAGE="$2"; shift 2 ;;
    --lora)        export GEN_MODE=lora MATERIAL_CKPT="${WEIGHTS_DIR}/videoneumat/t2v_lora/videoneumat-t2v-lora.safetensors"; shift ;;
    --name)        RUNNAME="$2"; shift 2 ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)             PROMPT="$1"; shift ;;
  esac
done
if [ -z "$PROMPT" ] && [ -z "$PROMPTS_CSV_IN" ]; then
  echo "ERROR: give a prompt (add --i2v IMAGE for image->video), or --prompts_csv. Try --help." >&2; exit 1
fi
if [ -n "$INPUT_IMAGE" ] && [ ! -f "$INPUT_IMAGE" ]; then
  echo "ERROR: --i2v image not found: $INPUT_IMAGE" >&2; exit 1
fi
if [ -n "$INPUT_IMAGE" ]; then
  INPUT_IMAGE="$(realpath -e -- "$INPUT_IMAGE")"
fi

# ---- run dir + split name (single token, no underscores) -------------------
STAMP="$(date +%Y%m%d%H%M%S)"
SPLIT="rel${STAMP}"
[ -n "$RUNNAME" ] && SPLIT="rel$(echo "$RUNNAME" | tr -cd '[:alnum:]')${STAMP}"
RUN_DIR="${NEUGEN_REPO}/outputs/release_runs/${SPLIT}"
GEN_OUTDIR="${RUN_DIR}/videos"
mkdir -p "$GEN_OUTDIR"
export RUN_DIR SPLIT GEN_OUTDIR
echo "run:      $SPLIT   ($([ -n "$INPUT_IMAGE" ] && echo 'I2V' || echo "T2V ${GEN_MODE}"))"
echo "run dir:  $RUN_DIR"

# ---- Stage A: generate the material video ----------------------------------
if [ -n "$INPUT_IMAGE" ]; then
  export INPUT_IMAGE PROMPT
  JID_GEN=$(sbatch --parsable --account "$ACCOUNT" --partition "$PARTITION" \
    --output="${RUN_DIR}/gen_%j.out" --error="${RUN_DIR}/gen_%j.err" --export=ALL \
    "${RELEASE_DIR}/gen_i2v.sbatch")
  echo "Stage A (i2v gen) : job $JID_GEN   (~70 min)"
else
  PROMPTS_CSV="${RUN_DIR}/prompts.csv"
  if [ -n "$PROMPTS_CSV_IN" ]; then
    cp "$PROMPTS_CSV_IN" "$PROMPTS_CSV"
  else
    printf 'prompt\n' > "$PROMPTS_CSV"
    printf '%s\n' "\"${PROMPT//\"/\"\"}\"" >> "$PROMPTS_CSV"
  fi
  N=$(python3 -c "import csv; print(sum(1 for r in csv.DictReader(open('$PROMPTS_CSV')) if (r.get('prompt') or '').strip()))")
  [ "$N" -ge 1 ] || { echo "ERROR: no prompts in $PROMPTS_CSV" >&2; exit 1; }
  export PROMPTS_CSV
  echo "prompts:  $N   ($PROMPTS_CSV)"
  JID_GEN=$(sbatch --parsable --account "$ACCOUNT" --partition "$PARTITION" \
    --array="0-$((N-1))" \
    --output="${RUN_DIR}/gen_%A_%a.out" --error="${RUN_DIR}/gen_%A_%a.err" --export=ALL \
    "${RELEASE_DIR}/gen_video.sbatch")
  echo "Stage A (gen)     : job $JID_GEN   (array 0-$((N-1)), ~70 min/video)"
fi

# ---- Stage B: extract the neural material (waits for gen) -------------------
JID_EXT=$(sbatch --parsable --account "$ACCOUNT" --partition "$PARTITION" \
  --dependency="afterok:${JID_GEN}" \
  --output="${RUN_DIR}/extract_%j.out" --error="${RUN_DIR}/extract_%j.err" --export=ALL \
  "${RELEASE_DIR}/extract_material.sbatch")
echo "Stage B (extract) : job $JID_EXT   (starts after gen succeeds)"

echo
echo "watch:   squeue -j ${JID_GEN},${JID_EXT}"
echo "results: ${RUN_DIR}/infer/final/${SPLIT}_split/sparse_traj/*/val_views.mp4"
