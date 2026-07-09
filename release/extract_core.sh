#!/bin/bash
# Stage B core (env-agnostic): build a dataset from the generated mp4(s), cache VAE
# latents, and run the LRM (optimize=False) to extract + render the neural material.
# Set NEUGEN_PY to the python to use (the container's, or a local conda env with the
# NeuGen deps). Needs, from config.env + the caller:
#   NEUGEN_REPO RELEASE_DIR RUN_DIR SPLIT GEN_OUTDIR WAN_MODELS LRM_LATENT LRM_MLP
set -euo pipefail
: "${NEUGEN_PY:?set NEUGEN_PY to a python with the NeuGen deps}"
: "${RELEASE_DIR:?}" ; : "${NEUGEN_REPO:?}" ; : "${RUN_DIR:?}" ; : "${SPLIT:?}" ; : "${GEN_OUTDIR:?}"

cd "$NEUGEN_REPO"
DSET="data/diffusion/${SPLIT}/sparse"

# (0) wire ./data for a fresh checkout (models + null_text_emb + scene states)
bash "$RELEASE_DIR/setup_data.sh"

# (1) bridge: generated mp4(s) -> NeuGen diffusion dataset
echo "[extract] building dataset $DSET from $GEN_OUTDIR"
rm -rf "data/diffusion/${SPLIT}"
mkdir -p "$DSET"
cp -r "$NEUGEN_REPO/assets/scene_state/sparse/0_scene_state" "$DSET/0_scene_state"
printf 'video,prompt\n' > "$DSET/metadata_clean.csv"
i=0 ; shopt -s nullglob
for mp4 in "$GEN_OUTDIR"/*.mp4; do
  name=$(printf 'mat%03d' "$i")
  mkdir -p "$DSET/$name"
  ln -sfn "$mp4" "$DSET/$name/btf_sequence.mp4"
  printf '%s/btf_sequence.mp4,""\n' "$name" >> "$DSET/metadata_clean.csv"
  i=$((i+1))
done
shopt -u nullglob
[ "$i" -gt 0 ] || { echo "[extract] ERROR: no .mp4 found in $GEN_OUTDIR"; exit 1; }
echo "[extract] dataset has $i material(s)"

# (2) cache VAE latents
echo "[extract] caching VAE latents for diffusion_${SPLIT}_sparse_all"
"$NEUGEN_PY" src/cache_vae_latents.py \
  "dataset=diffusion_${SPLIT}_sparse_all" resolution=1024 batch_size=8 \
  "out_root=${RUN_DIR}/cache"
mkdir -p "${DSET}/0 latents"
cp -r "${RUN_DIR}"/cache/latents/*/ "${DSET}/0 latents/"

# (3) LRM inference (optimize=False)
echo "[extract] LRM inference"
"$NEUGEN_PY" src/inverse_render_LRM.py \
  "train_dataset=[diffusion_${SPLIT}_sparse_all]" \
  "val_dataset=[diffusion_${SPLIT}_sparse_all]" \
  optimize=False date=inference time=final \
  "checkpoint.latent=${LRM_LATENT}" "checkpoint.network=${LRM_MLP}" \
  latent/upsampler=vae network.min_z=0.64 network.max_offset=0.02 resolution=1024 \
  "out_root=${RUN_DIR}/infer"

echo "[extract] DONE. Results: ${RUN_DIR}/infer/final/${SPLIT}_split/sparse_traj/*/{val_views,train_views}.mp4"
